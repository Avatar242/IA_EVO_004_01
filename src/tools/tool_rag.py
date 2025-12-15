# src/tools/tool_rag.py

import hashlib
import datetime
from typing import List, Dict, Any
from .base_tool import BaseTool
from services.base_api_client import BaseApiClient
from core.vector_db_manager import VectorDBManager
from core.document_processor import DocumentProcessor

class RAGTool(BaseTool):
    """
    Herramienta para la Búsqueda y Generación Aumentada (Retrieval-Augmented Generation - RAG).
    Permite al agente indexar documentos en una base de datos vectorial y luego usar esa
    base de datos para responder preguntas basándose en el contenido de los documentos.
    """

    def __init__(self, api_client: BaseApiClient, db_manager: VectorDBManager, doc_processor: DocumentProcessor):
        """
        Inicializa la RAGTool con sus dependencias.

        Args:
            api_client: Cliente para generar embeddings y respuestas.
            db_manager: Gestor de la base de datos vectorial.
            doc_processor: Procesador para leer y dividir documentos.
        """
        self._api_client = api_client
        self._db_manager = db_manager
        self._doc_processor = doc_processor
        print("RAGTool inicializada.")

    @property
    def name(self) -> str:
        return "rag_tool"

    @property
    def description(self) -> str:
        return (
            "Responde preguntas usando una base de conocimiento de documentos previamente indexados. "
            "Úsala cuando el usuario pregunte sobre contenido específico de un archivo. "
            "La entrada para esta herramienta debe ser un JSON con las claves 'mode' ('query') y 'user_query' (la pregunta del usuario)."
        )

    def execute(self, mode: str, **kwargs: Any) -> str:
        """
        Punto de entrada principal para la RAGTool.

        Args:
            mode (str): El modo de operación ('index' o 'query').
            **kwargs: Argumentos adicionales dependiendo del modo.
                - para 'index': file_path, category, tags
                - para 'query': user_query

        Returns:
            str: El resultado de la operación.
        """
        if mode == "index":
            return self.index_document(
                file_path=kwargs.get("file_path"),
                category=kwargs.get("category", "general"),
                tags=kwargs.get("tags", [])
            )
        elif mode == "query":
            return self._query_rag(query=kwargs.get("user_query"))
        else:
            return f"Modo '{mode}' no reconocido para RAGTool. Use 'index' o 'query'."

    def index_document(self, file_path: str, category: str, tags: List[str]) -> str:
        """
        Procesa, vectoriza e indexa el contenido de un documento en la BD vectorial.
        """
        try:
            # 1. Procesar el documento para obtener los trozos de texto
            chunks = self._doc_processor.process_pdf(file_path)
            if not chunks:
                return "No se pudo extraer texto del documento o el documento está vacío."

            # 2. Generar embeddings para cada trozo
            print(f"Generando embeddings para {len(chunks)} trozos...")
            embeddings = [self._api_client.generate_embeddings(chunk) for chunk in chunks]
            
            # 3. Preparar IDs y metadata para cada trozo
            ids = []
            metadatas = []
            timestamp = datetime.datetime.utcnow().isoformat()
            
            for i, chunk in enumerate(chunks):
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
                doc_id = f"{file_path}_{i}"
                ids.append(doc_id)
                
                metadatas.append({
                    "source_id": file_path,
                    "document_type": "pdf",
                    "chunk_seq_id": i,
                    "text_hash": chunk_hash,
                    "category": category,
                    # Convertimos la lista de tags en un solo string separado por comas
                    "tags": ",".join(tags),
                    "created_at": timestamp
                })

            # 4. Añadir todo a la base de datos vectorial
            success = self._db_manager.add_documents(ids, chunks, embeddings, metadatas)
            if success:
                return f"Documento '{file_path}' indexado exitosamente con {len(chunks)} trozos."
            else:
                return f"Hubo un error al indexar el documento '{file_path}'."

        except Exception as e:
            return f"Ocurrió un error inesperado durante la indexación: {e}"

    def _query_rag(self, query: str) -> str:
        """
        Realiza una consulta RAG: busca contexto relevante y genera una respuesta.
        """
        if not query:
            return "La consulta no puede estar vacía."
            
        print(f"Realizando búsqueda RAG para la consulta: '{query}'")
        
        # 1. Generar embedding para la consulta del usuario
        query_embedding = self._api_client.generate_embeddings(query)
        if not query_embedding:
            return "No se pudo generar el embedding para la consulta."

        # 2. Buscar en la base de datos los trozos de contexto más relevantes
        search_results = self._db_manager.query(query_embedding, n_results=3)
        if not search_results:
            return "No se encontró información relevante en la base de conocimiento para responder a tu pregunta."

        # 3. Construir el contexto y el prompt aumentado
        context = "\n\n---\n\n".join([result['document'] for result in search_results])
        
        rag_prompt = (
            f"Basándote únicamente en el siguiente CONTEXTO EXTRAÍDO de documentos, responde a la PREGUNTA del usuario. "
            f"Si el contexto no contiene la respuesta, di explícitamente que no tienes suficiente información.\n\n"
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA:\n{query}"
        )

        # 4. Generar la respuesta final usando el LLM
        print("Generando respuesta final basada en el contexto recuperado...")
        final_answer = self._api_client.generate_content(prompt=rag_prompt)
        
        return final_answer