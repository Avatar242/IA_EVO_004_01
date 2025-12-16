# src/core/document_processor.py

from typing import List
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """
    MODIFICADO: Procesa documentos usando estrategias avanzadas de división de texto (chunking).
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        """
        Inicializa el procesador de documentos.

        Args:
            chunk_size (int): El tamaño máximo de cada trozo (el splitter intentará respetarlo).
            chunk_overlap (int): El número de caracteres que se superpondrán entre trozos.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        print(f"DocumentProcessor inicializado con RecursiveCharacterTextSplitter (chunk_size={chunk_size}, chunk_overlap={chunk_overlap}).")

    def process_pdf(self, file_path: str) -> List[str]:
        """
        MODIFICADO: Lee un PDF página por página, dividiendo el texto de cada página
        individualmente para respetar los límites estructurales del documento.
        """
        print(f"Procesando el archivo PDF: {file_path}")
        try:
            reader = pypdf.PdfReader(file_path)
            all_chunks = []
            total_chars = 0

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if not page_text:
                    continue
                
                total_chars += len(page_text)
                
                # Dividimos el texto de la página actual
                page_chunks = self.text_splitter.split_text(page_text)
                
                # AÑADIDO: Podríamos enriquecer los chunks con metadata de la página si quisiéramos.
                # Por ahora, simplemente los agregamos a la lista total.
                all_chunks.extend(page_chunks)

            print(f"Texto extraído: {total_chars} caracteres.")
            print(f"Texto dividido en {len(all_chunks)} trozos (chunks) procesando página por página.")
            return all_chunks
            
        except FileNotFoundError:
            print(f"[ERROR] Archivo no encontrado en: {file_path}")
            raise
        except Exception as e:
            print(f"[ERROR] No se pudo leer el archivo PDF: {e}")
            raise