# src/core/document_processor.py

from typing import List
import pypdf

# NUEVO: Importamos el divisor de texto de Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """
    MODIFICADO: Procesa documentos usando estrategias avanzadas de división de texto (chunking).
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicializa el procesador de documentos.

        Args:
            chunk_size (int): El tamaño máximo de cada trozo (el splitter intentará respetarlo).
            chunk_overlap (int): El número de caracteres que se superpondrán entre trozos.
        """
        # MODIFICADO: Creamos una instancia del splitter de Langchain.
        # Intenta dividir primero por párrafos, luego por saltos de línea, etc.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        print(f"DocumentProcessor inicializado con RecursiveCharacterTextSplitter (chunk_size={chunk_size}, chunk_overlap={chunk_overlap}).")

    def process_pdf(self, file_path: str) -> List[str]:
        """
        Lee un archivo PDF, extrae el texto y lo divide usando el text splitter.
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

            # MODIFICADO: Usamos el splitter para dividir el texto
            chunks = self.text_splitter.split_text(full_text)
            print(f"Texto dividido en {len(chunks)} trozos (chunks).")
            return chunks
            
        except FileNotFoundError:
            print(f"[ERROR] Archivo no encontrado en: {file_path}")
            raise
        except Exception as e:
            print(f"[ERROR] No se pudo leer el archivo PDF: {e}")
            raise

    # El método _chunk_text manual ya no es necesario y ha sido eliminado.