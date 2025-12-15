# src/core/vector_db_manager.py

from typing import List, Dict, Any
import chromadb

class VectorDBManager:
    """
    Gestiona todas las interacciones con la base de datos vectorial (ChromaDB).

    Esta clase abstrae la lógica de conexión, almacenamiento y consulta de la base de datos,
    permitiendo que el resto de la aplicación interactúe con el almacenamiento de vectores
    de una manera simple y consistente.
    """

    def __init__(self, db_path: str = "db", collection_name: str = "main_collection"):
        """
        Inicializa el gestor de la base de datos.

        Args:
            db_path (str): La ruta a la carpeta donde se almacenará la base de datos.
            collection_name (str): El nombre de la colección a utilizar dentro de la BD.
        """
        try:
            # Crea un cliente persistente que guarda los datos en el disco
            self.client = chromadb.PersistentClient(path=db_path)
            
            # Obtiene la colección. Si no existe, la crea automáticamente.
            self.collection = self.client.get_or_create_collection(name=collection_name)
            
            print(f"VectorDBManager inicializado. Conectado a la colección '{collection_name}' en '{db_path}'.")
            print(f"Documentos actuales en la colección: {self.collection.count()}")
            
        except Exception as e:
            print(f"[ERROR CRÍTICO] No se pudo inicializar ChromaDB: {e}")
            raise

    def add_documents(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Añade documentos, junto con sus embeddings y metadata, a la colección.

        Args:
            ids (List[str]): Una lista de IDs únicos para cada documento/chunk.
            documents (List[str]): La lista de textos de los documentos/chunks.
            embeddings (List[List[float]]): La lista de vectores de embedding correspondientes.
            metadatas (List[Dict[str, Any]]): La lista de diccionarios de metadata.
        
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario.
        """
        try:
            # ChromaDB puede manejar la inserción de lotes de documentos de una sola vez
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Se han añadido {len(ids)} documentos a la colección '{self.collection.name}'. Total: {self.collection.count()}")
            return True
        except Exception as e:
            print(f"[ERROR] No se pudieron añadir los documentos a ChromaDB: {e}")
            return False

    def query(self, query_embedding: List[float], n_results: int = 5, where_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        MODIFICADO: Realiza una búsqueda de similitud, con un filtro de metadata opcional.

        Args:
            query_embedding (List[float]): El vector de embedding de la consulta.
            n_results (int): El número de resultados a devolver.
            where_filter (Dict[str, Any], optional): Un diccionario para filtrar la metadata.
                                                     Ejemplo: {"category": "ciberseguridad"}

        Returns:
            List[Dict[str, Any]]: Una lista de resultados.
        """
        try:
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }
            if where_filter:
                query_params["where"] = where_filter
                print(f"Ejecutando consulta con filtro: {where_filter}")
            else:
                print("Ejecutando consulta sin filtro.")

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
        except Exception as e:
            print(f"[ERROR] Ocurrió un error al consultar ChromaDB: {e}")
            return [] 



