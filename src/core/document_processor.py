# src/core/document_processor.py

from typing import List
import pypdf

class DocumentProcessor:
    """
    Una clase dedicada a procesar documentos para la funcionalidad RAG.

    Actualmente, se enfoca en leer archivos PDF, extraer su texto y dividirlo
    en trozos manejables (chunks) para su posterior procesamiento y embedding.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicializa el procesador de documentos.

        Args:
            chunk_size (int): El tamaño deseado para cada trozo de texto (en caracteres).
            chunk_overlap (int): El número de caracteres que se superpondrán entre
                                 trozos consecutivos para mantener el contexto.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size.")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print("DocumentProcessor inicializado.")

    def process_pdf(self, file_path: str) -> List[str]:
        """
        Lee un archivo PDF, extrae el texto y lo divide en trozos.

        Args:
            file_path (str): La ruta al archivo PDF.

        Returns:
            List[str]: Una lista de trozos de texto extraídos del documento.
        
        Raises:
            FileNotFoundError: Si el archivo no se encuentra en la ruta especificada.
            Exception: Para otros errores relacionados con la lectura del PDF.
        """
        print(f"Procesando el archivo PDF: {file_path}")
        try:
            reader = pypdf.PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            
            print(f"Texto extraído: {len(full_text)} caracteres.")
            
            if not full_text:
                return []

            return self._chunk_text(full_text)
            
        except FileNotFoundError:
            print(f"[ERROR] Archivo no encontrado en: {file_path}")
            raise
        except Exception as e:
            print(f"[ERROR] No se pudo leer el archivo PDF: {e}")
            raise

    def _chunk_text(self, text: str) -> List[str]:
        """
        Divide un texto largo en trozos más pequeños con superposición.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            
            # El siguiente trozo comienza antes del final del actual para crear la superposición
            start += self.chunk_size - self.chunk_overlap
            
        print(f"Texto dividido en {len(chunks)} trozos.")
        return chunks