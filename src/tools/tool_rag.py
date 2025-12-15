# src/tools/tool_rag.py

import hashlib
import datetime
import json
import re
from typing import List, Dict, Any, Tuple
from .base_tool import BaseTool
from services.base_api_client import BaseApiClient
from core.vector_db_manager import VectorDBManager
from core.document_processor import DocumentProcessor

class RAGTool(BaseTool):
    """
    Herramienta para la Búsqueda y Generación Aumentada (RAG).
    Permite al agente indexar documentos de forma inteligente y responder preguntas
    basándose en el contenido de esos documentos.
    """

    def __init__(self, api_client: BaseApiClient, db_manager: VectorDBManager, doc_processor: DocumentProcessor):
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
        if mode == "index":
            # La llamada ahora es más simple, solo necesita el file_path
            return self.index_document(file_path=kwargs.get("file_path"))
        elif mode == "query":
            return self._query_rag(query=kwargs.get("user_query"))
        else:
            return f"Modo '{mode}' no reconocido para RAGTool. Use 'index' o 'query'."

    def _get_document_category(self, text_excerpt: str) -> Tuple[str, List[str]]:
        """
        Usa el LLM para determinar automáticamente la categoría y las etiquetas de un documento.
        """
        print("Determinando la categoría del documento usando el LLM...")
        
        # SOLUCIÓN 2: Prompt de categorización mejorado
        system_prompt = (
            "Tu rol es ser un experto bibliotecario. Analiza el siguiente extracto de texto. "
            "Tu única tarea es devolver un objeto JSON con dos claves: "
            "1. 'category': una única palabra específica y descriptiva en minúsculas (ej: 'finanzas', 'ciberseguridad', 'salud'). "
            "2. 'tags': una lista de hasta 5 términos técnicos o entidades clave, muy específicos del texto, en minúsculas. Evita palabras genéricas como 'información' o 'documento'. "
            "No añadas explicaciones. Tu respuesta debe ser solo el JSON."
        )
        
        categorization_prompt = f"Extracto del documento:\n\n{text_excerpt}"

        try:
            response_str = self._api_client.generate_content(
                prompt=categorization_prompt,
                history=[{'role': 'system', 'content': system_prompt}]
            )
            
            json_match = re.search(r"\{.*\}", response_str, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON object found in LLM response", response_str, 0)
            
            clean_json_str = json_match.group(0)
            data = json.loads(clean_json_str)
            
            category = data.get("category", "general")
            tags = data.get("tags", [])
            
            print(f"Categoría determinada: '{category}', Tags: {tags}")
            return category, tags

        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo determinar la categoría automáticamente: {e}. Usando valores por defecto.")
            return "general", []

    def index_document(self, file_path: str) -> str:
        """
        MODIFICADO: Procesa, categoriza, vectoriza e indexa un documento.
        """
        try:
            chunks = self._doc_processor.process_pdf(file_path)
            if not chunks:
                return "No se pudo extraer texto del documento."

            # NUEVO: Usar los primeros 2000 caracteres para la categorización
            document_excerpt = " ".join(chunks)[:2000]
            category, tags = self._get_document_category(document_excerpt)

            print(f"Generando embeddings para {len(chunks)} trozos...")
            embeddings = [self._api_client.generate_embeddings(chunk) for chunk in chunks]
            
            ids, metadatas = [], []
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
                    "category": category, # Usamos el valor autogenerado
                    "tags": ",".join(tags), # Usamos los valores autogenerados
                    "created_at": timestamp
                })

            success = self._db_manager.add_documents(ids, chunks, embeddings, metadatas)
            if success:
                return f"Documento '{file_path}' indexado exitosamente en la categoría '{category}' con {len(chunks)} trozos."
            else:
                return f"Hubo un error al indexar el documento '{file_path}'."

        except Exception as e:
            return f"Ocurrió un error inesperado durante la indexación: {e}"

    def _query_rag(self, query: str) -> str:
        # Este método no necesita cambios
        if not query:
            return "La consulta no puede estar vacía."
        print(f"Realizando búsqueda RAG para la consulta: '{query}'")
        query_embedding = self._api_client.generate_embeddings(query)
        if not query_embedding:
            return "No se pudo generar el embedding para la consulta."
        search_results = self._db_manager.query(query_embedding, n_results=3)
        if not search_results:
            return "No se encontró información relevante en la base de conocimiento para responder a tu pregunta."
        context = "\n\n---\n\n".join([result['document'] for result in search_results])
        rag_prompt = (
            f"Basándote únicamente en el siguiente CONTEXTO EXTRAÍDO de documentos, responde a la PREGUNTA del usuario. "
            f"Si el contexto no contiene la respuesta, di explícitamente que no tienes suficiente información.\n\n"
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA:\n{query}"
        )
        print("Generando respuesta final basada en el contexto recuperado...")
        final_answer = self._api_client.generate_content(prompt=rag_prompt)
        return final_answer