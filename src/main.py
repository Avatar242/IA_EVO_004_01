# src/main.py

import os
import sys
from dotenv import load_dotenv

# --- Configuración Inicial ---
# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Añadimos la carpeta 'src' al path de Python para permitir importaciones absolutas
# Esto es crucial para que nuestro script pueda encontrar los módulos en 'core', 'services', etc.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.dispatcher import Dispatcher
from core.tool_registry import ToolRegistry
from services.ollama_api_client import OllamaApiClient
from services.gemini_api_client import GeminiApiClient
from tools.general_conversation_tool import GeneralConversationTool
# (Aquí importaríamos futuras herramientas, como from tools.tool_rag import RAGTool)

def main():
    """
    Punto de entrada principal y orquestador de la aplicación del agente.
    """
    print("Iniciando el Agente de IA...")

    # --- 1. Composition Root (Raíz de Composición) ---
    # Aquí es donde se "construye" la aplicación, instanciando y conectando todos los componentes.

    # Seleccionar e inicializar el cliente de API basado en la configuración del entorno
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
        sys.exit(1) # Salir si la configuración esencial falla

    # Inicializar el registro de herramientas
    tool_registry = ToolRegistry()

    # Inicializar y registrar las herramientas disponibles
    # Inyectamos el api_client en las herramientas que lo necesiten
    general_tool = GeneralConversationTool(api_client=api_client)
    tool_registry.register_tool(general_tool)
    
    # (Aquí registraríamos futuras herramientas:
    # rag_tool = RAGTool(api_client=api_client)
    # tool_registry.register_tool(rag_tool))

    # Inicializar el despachador (Dispatcher)
    dispatcher = Dispatcher(api_client=api_client)

    # --- 2. Bucle Principal de la Aplicación ---
    print("\nAgente listo. Escribe 'salir' para terminar.")
    
    conversation_history = []

    while True:
        try:
            user_input = input("\nTú: ")
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("Agente: Adiós.")
                break

            # 3. Despacho (Dispatch)
            # El dispatcher elige qué herramienta usar
            tool_name, tool_args = dispatcher.dispatch(user_input, conversation_history, tool_registry)

            # 4. Ejecución
            # Obtenemos la herramienta del registro y la ejecutamos
            tool = tool_registry.get_tool(tool_name)
            response = tool.execute(user_prompt=user_input, history=conversation_history, **tool_args)

            print(f"Agente: {response}")

            # 5. Mantenimiento del Historial
            # Actualizamos el historial para mantener el contexto
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nAgente: Adiós.")
            break
        except Exception as e:
            print(f"\n[ERROR INESPERADO] Ocurrió un error: {e}")


if __name__ == "__main__":
    main()