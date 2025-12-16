# src/core/vector_db_manager.py

from typing import List, Dict, Any
import chromadb
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Asegúrate de haber descargado los recursos de NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("[ADVERTENCIA] Faltan recursos de NLTK. Ejecuta `python setup_nltk.py`.")

class VectorDBManager:
    """
    MODIFICADO: Gestiona una búsqueda híbrida combinando ChromaDB (semántica) y un índice BM25 (palabras clave).
    """

    def __init__(self, db_path: str = "db", collection_name: str = "main_collection"):
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            
            ### NUEVO: Atributos para el índice BM25
            self.bm25_index = None
            self.documents_cache = {}  # Almacena {id: {'document': str, 'metadata': dict}}
            self.id_corpus = []        # Mantiene el orden de los IDs para el mapeo con BM25

            # Construir el índice BM25 con los datos existentes en ChromaDB
            self._build_bm25_index_from_db()

            print(f"VectorDBManager (Híbrido) inicializado. Conectado a '{collection_name}'.")
            print(f"Documentos en ChromaDB: {self.collection.count()}. Documentos en índice BM25: {len(self.id_corpus)}.")
            
        except Exception as e:
            print(f"[ERROR CRÍTICO] No se pudo inicializar ChromaDB o BM25: {e}")
            raise

    ### NUEVO: Reconstruye el índice BM25 en memoria a partir de ChromaDB
    def _build_bm25_index_from_db(self):
        print("Construyendo índice BM25 desde la base de datos...")
        # Obtenemos TODOS los documentos de la colección
        # Nota: Esto podría ser ineficiente para colecciones masivas (+100k docs)
        existing_docs = self.collection.get(include=["metadatas", "documents"])
        
        if not existing_docs or not existing_docs['ids']:
            print("La base de datos está vacía. No se construyó el índice BM25.")
            return

        # Limpiamos y preparamos los datos
        self.documents_cache = {}
        self.id_corpus = existing_docs['ids']
        documents_list = existing_docs['documents']

        for i, doc_id in enumerate(self.id_corpus):
            self.documents_cache[doc_id] = {
                'document': documents_list[i],
                'metadata': existing_docs['metadatas'][i]
            }
        
        # Preparamos el corpus para BM25
        stop_words = set(stopwords.words('english')) # Se puede adaptar a otros idiomas
        tokenized_corpus = [
            [word for word in word_tokenize(doc.lower()) if word.isalnum() and word not in stop_words]
            for doc in documents_list
        ]
        
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"Índice BM25 construido con {len(self.id_corpus)} documentos.")

    def add_documents(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        try:
            self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            
            ### NUEVO: Reconstruimos el índice BM25 para incluir los nuevos documentos
            print(f"Se han añadido {len(ids)} documentos. Reconstruyendo el índice BM25 para mantener la consistencia...")
            self._build_bm25_index_from_db()
            
            return True
        except Exception as e:
            print(f"[ERROR] No se pudieron añadir los documentos: {e}")
            return False

    ### NUEVO: Método principal para la búsqueda híbrida
    def hybrid_search(self, query_text: str, query_embedding: List[float], n_results: int = 10, where_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        all_results = {}

        # 1. Búsqueda semántica (Vectorial)
        try:
            semantic_results = self._vector_search(query_embedding, n_results, where_filter)
            for res in semantic_results:
                all_results[res['id']] = res
            print(f"Búsqueda semántica encontró {len(semantic_results)} resultados.")
        except Exception as e:
            print(f"[ERROR] en búsqueda semántica: {e}")

        # 2. Búsqueda por palabras clave (BM25)
        # Nota: El filtro 'where' no se puede aplicar a BM25 de forma sencilla. Es una limitación.
        if self.bm25_index:
            try:
                keyword_results = self._keyword_search(query_text, n_results)
                print(f"Búsqueda por palabra clave encontró {len(keyword_results)} resultados.")
                for res in keyword_results:
                    if res['id'] not in all_results: # Evitar sobreescribir si ya existe
                        all_results[res['id']] = res
            except Exception as e:
                print(f"[ERROR] en búsqueda por palabra clave: {e}")

        # Devolvemos una lista combinada y sin duplicados
        return list(all_results.values())[:n_results]

    ### MODIFICADO: El método 'query' ahora es privado y renombrado
    def _vector_search(self, query_embedding: List[float], n_results: int, where_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        query_params = {"query_embeddings": [query_embedding], "n_results": n_results}
        if where_filter:
            query_params["where"] = where_filter
        
        results = self.collection.query(**query_params)
        
        combined_results = []
        if results and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                combined_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
        return combined_results

    ### NUEVO: Método para la búsqueda por palabras clave
    def _keyword_search(self, query_text: str, n_results: int) -> List[Dict[str, Any]]:
        stop_words = set(stopwords.words('english'))
        tokenized_query = [word for word in word_tokenize(query_text.lower()) if word.isalnum() and word not in stop_words]
        
        # Obtenemos los scores de los documentos
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Obtenemos los N mejores índices
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n_results]
        
        # Construimos los resultados a partir de los índices
        results = []
        for i in top_n_indices:
            doc_id = self.id_corpus[i]
            if doc_scores[i] > 0: # Solo incluimos resultados con score positivo
                result_doc = self.documents_cache.get(doc_id, {})
                results.append({
                    'id': doc_id,
                    'document': result_doc.get('document'),
                    'metadata': result_doc.get('metadata'),
                    'score_bm25': doc_scores[i]
                })
        return results