# src/main.py

import os
import sys
from dotenv import load_dotenv

# --- Configuración Inicial ---
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Importaciones de Componentes ---
from core.dispatcher import Dispatcher
from core.tool_registry import ToolRegistry
from services.ollama_api_client import OllamaApiClient
from services.gemini_api_client import GeminiApiClient
from tools.general_conversation_tool import GeneralConversationTool
from core.document_processor import DocumentProcessor
from core.vector_db_manager import VectorDBManager
from tools.tool_rag import RAGTool

def main():
    """
    Punto de entrada principal y orquestador de la aplicación del agente.
    """
    print("Iniciando el Agente de IA...")

    # --- 1. Composition Root ---
    api_provider = os.getenv("AI_PROVIDER", "ollama").lower()
    api_client = None
    try:
        if api_provider == "gemini":
            api_client = GeminiApiClient()
        elif api_provider == "ollama":
            api_client = OllamaApiClient()
        else:
            raise ValueError(f"Proveedor de IA no válido: {api_provider}. Opciones válidas: 'gemini', 'ollama'.")
    except (ValueError, RuntimeError) as e:
        print(f"[ERROR CRÍTICO] No se pudo inicializar el cliente de API: {e}")
        sys.exit(1)

    db_manager = VectorDBManager(collection_name=f"{api_provider}_collection")
    doc_processor = DocumentProcessor(chunk_size=1024, chunk_overlap=200) # Usamos el chunking mejorado
    
    tool_registry = ToolRegistry()

    general_tool = GeneralConversationTool(api_client=api_client)
    tool_registry.register_tool(general_tool)
    
    rag_tool = RAGTool(api_client=api_client, db_manager=db_manager, doc_processor=doc_processor)
    tool_registry.register_tool(rag_tool)

    dispatcher = Dispatcher(api_client=api_client)

    # --- 2. Bucle Principal de la Aplicación ---
    print("\nAgente listo. Comandos especiales: !index <ruta_pdf>, !query <pregunta>, salir")
    
    conversation_history = []

    while True:
        try:
            user_input = input("\nTú: ").strip() # Usamos .strip() para eliminar espacios en blanco
            
            # SOLUCIÓN 1: Ignorar entradas vacías
            if not user_input:
                continue

            if user_input.lower() in ["salir", "exit", "quit"]:
                print("Agente: Adiós.")
                break

            response = ""
            
            if user_input.startswith("!index "):
                file_path = user_input.split(" ", 1)[1]
                response = rag_tool.execute(mode="index", file_path=file_path)
            elif user_input.startswith("!query "):
                query = user_input.split(" ", 1)[1]
                response = rag_tool.execute(mode="query", user_query=query)
            else:
                # --- Flujo normal del despachador ---
                # El dispatcher ahora es responsable de elegir la herramienta Y preparar los argumentos
                tool_name, tool_args = dispatcher.dispatch(user_input, conversation_history, tool_registry)

                # Ejecutamos la herramienta con los argumentos que el dispatcher preparó
                tool = tool_registry.get_tool(tool_name)
                response = tool.execute(**tool_args)

            print(f"Agente: {response}")

            if not user_input.startswith("!index "):
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nAgente: Adiós.")
            break
        except Exception as e:
            print(f"\n[ERROR INESPERADO] Ocurrió un error: {e}")

if __name__ == "__main__":
    main()