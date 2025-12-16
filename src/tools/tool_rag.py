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
            "Indispensable para responder preguntas sobre el contenido específico de documentos, informes o archivos que han sido previamente indexados. "
            "Úsala siempre que la pregunta del usuario haga referencia a un documento, un informe, un archivo, o un tema muy específico que probablemente no sea de conocimiento general."
        )

    def execute(self, mode: str, **kwargs: Any) -> str:
        if mode == "index":
            return self.index_document(file_path=kwargs.get("file_path"))
        elif mode == "query":
            user_query = kwargs.get("user_query")
            where_filter = kwargs.get("where_filter")
            return self._intelligent_query_rag(query=user_query, where_filter=where_filter)
        else:
            return f"Modo '{mode}' no reconocido para RAGTool. Use 'index' o 'query'."

    # ... (los métodos _get_document_category e index_document no cambian) ...
    def _get_document_category(self, text_excerpt: str) -> Tuple[str, List[str]]:
        """
        Usa el LLM para determinar automáticamente la categoría y las etiquetas de un documento.
        """
        print("Determinando la categoría del documento usando el LLM...")
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
        try:
            chunks = self._doc_processor.process_pdf(file_path)
            if not chunks:
                return "No se pudo extraer texto del documento."

            document_excerpt = " ".join(chunks)[:2000]
            category, tags = self._get_document_category(document_excerpt)

            embeddings = []
            print(f"Generando embeddings finales para {len(chunks)} chunks...")
            for i, chunk in enumerate(chunks):
                embedding = self._api_client.generate_embeddings(chunk)
                if not embedding:
                    raise Exception(f"Fallo crítico al generar embedding para el chunk {i}.")
                embeddings.append(embedding)
                print(f"Generando embedding {i+1}/{len(chunks)}...", end="\r")
            print("\nGeneración de embeddings finales completada.")

            ids = [f"{file_path}_{i}" for i in range(len(chunks))]
            metadatas = [{
                "source_id": file_path, "document_type": "pdf", "chunk_seq_id": i,
                "text_hash": hashlib.sha256(chunk.encode()).hexdigest(),
                "category": category, "tags": ",".join(tags),
                "created_at": datetime.datetime.utcnow().isoformat()
            } for i, chunk in enumerate(chunks)]

            success = self._db_manager.add_documents(ids, chunks, embeddings, metadatas)
            if success:
                return f"Documento '{file_path}' indexado exitosamente en la categoría '{category}' con {len(chunks)} trozos."
            else:
                return f"Hubo un error al indexar el documento '{file_path}'."

        except Exception as e:
            return f"Ocurrió un error inesperado durante la indexación: {e}"


    def _intelligent_query_rag(self, query: str, where_filter: Dict[str, Any] = None) -> str:
        if not query:
            return "La consulta no puede estar vacía."
        
        print(f"--- Iniciando Búsqueda RAG Inteligente para: '{query}' ---")
        
        # 1. Búsqueda inicial
        print("\n[Paso 1/3] Realizando búsqueda vectorial inicial...")
        search_results = self._perform_search(query, where_filter)

        # 2. Validación y posible reintento con COMBINACIÓN
        if not search_results or not self._validate_context(query, search_results):
            print("\n[Paso 2/3] Contexto inicial insuficiente. Intentando transformar y combinar...")
            
            enhanced_query = self._transform_query(query)
            if enhanced_query.lower() != query.lower():
                print(f"  -> Nueva consulta generada: '{enhanced_query}'")
                # Realizamos la segunda búsqueda
                second_results = self._perform_search(enhanced_query, where_filter)
                
                ### NUEVO: Lógica de combinación y de-duplicación
                if second_results:
                    print(f"  -> Combinando {len(search_results)} resultados iniciales con {len(second_results)} nuevos resultados.")
                    combined_results = search_results + second_results
                    
                    # Usamos un diccionario para eliminar duplicados basados en el 'id' del chunk
                    unique_results = {}
                    for result in combined_results:
                        unique_results[result['id']] = result
                    
                    search_results = list(unique_results.values())
                    print(f"  -> Contexto final combinado con {len(search_results)} chunks únicos.")

            else:
                print("  -> El LLM no pudo generar una consulta alternativa. Usando resultados originales.")

        # 3. Generación de respuesta final
        print("\n[Paso 3/3] Generando respuesta final basada en el mejor contexto disponible...")
        if not search_results:
            return "No se encontró información relevante en la base de conocimiento."

        return self._generate_final_answer(query, search_results)

    def _perform_search(self, query: str, where_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        query_embedding = self._api_client.generate_embeddings(query)
        if not query_embedding:
            print("[ADVERTENCIA] No se pudo generar embedding para la consulta.")
            return []
        
        return self._db_manager.query(query_embedding, n_results=10, where_filter=where_filter)

    def _validate_context(self, original_query: str, search_results: List[Dict[str, Any]]) -> bool:
        print("  -> Validando la calidad del contexto recuperado con el LLM...")
        if not search_results: # Si no hay resultados, la validación es False por definición
            return False
            
        context = "\n\n---\n\n".join([result['document'] for result in search_results])
        
        validation_prompt = (
            f"Pregunta del Usuario: '{original_query}'\n\n"
            f"Contexto Recuperado:\n---\n{context}\n---\n\n"
            "Analiza el contexto. ¿Contiene información que responda directa y completamente a la pregunta del usuario? "
            "Responde únicamente con la palabra 'SI' o 'NO'."
        )
        
        try:
            response = self._api_client.generate_content(validation_prompt).strip().upper()
            print(f"  -> Respuesta de validación del LLM: '{response}'")
            return "SI" in response
        except Exception as e:
            print(f"[ADVERTENCIA] Falló la validación del LLM: {e}. Asumiendo que el contexto es válido.")
            return True

    def _transform_query(self, original_query: str) -> str:
        transform_prompt = (
            "Tu tarea es reformular la siguiente pregunta de un usuario para hacerla más efectiva en una búsqueda de base de datos semántica. "
            "Enfócate en las palabras clave y la intención. No respondas a la pregunta, solo transfórmala.\n\n"
            f"Pregunta Original: '{original_query}'\n\n"
            "Pregunta Reformulada:"
        )
        try:
            return self._api_client.generate_content(transform_prompt).strip()
        except Exception as e:
            print(f"[ADVERTENCIA] Falló la transformación de la consulta: {e}. Usando la consulta original.")
            return original_query

    def _generate_final_answer(self, original_query: str, search_results: List[Dict[str, Any]]) -> str:
        context = "\n\n---\n\n".join([result['document'] for result in search_results])
        
        rag_prompt = (
            f"Basándote únicamente en el siguiente CONTEXTO EXTRAÍDO de documentos, responde a la PREGUNTA del usuario. "
            f"Si el contexto no contiene la respuesta, di explícitamente que no tienes suficiente información.\n\n"
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA:\n{original_query}"
        )

        return self._api_client.generate_content(prompt=rag_prompt)