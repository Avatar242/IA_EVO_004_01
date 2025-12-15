# src/core/dispatcher.py

import json
from services.base_api_client import BaseApiClient
from core.tool_registry import ToolRegistry

class Dispatcher:
    """
    El "cerebro" del agente. Decide qué herramienta usar para responder a una consulta.

    El Dispatcher utiliza un LLM para analizar la consulta del usuario y las
    especificaciones de las herramientas disponibles, y luego selecciona la herramienta
    más adecuada. No ejecuta la herramienta, solo determina el plan de ejecución.
    """

    def __init__(self, api_client: BaseApiClient):
        """
        Inicializa el Dispatcher con un cliente de API para comunicarse con el LLM.

        Args:
            api_client (BaseApiClient): El cliente de API (Ollama o Gemini) que se usará
                                        para el razonamiento del LLM.
        """
        self._api_client = api_client
        print("Dispatcher inicializado.")

    def _build_system_prompt(self, tool_registry: ToolRegistry) -> str:
        """
        Construye el prompt del sistema que instruye al LLM sobre cómo elegir una herramienta.
        """
        tool_specs = tool_registry.get_tool_specifications()
        # Convertimos las especificaciones a un formato JSON legible para el prompt
        tool_specs_json = json.dumps(tool_specs, indent=2)

        return f"""
                    Tu rol es ser un despachador inteligente. Tu objetivo es analizar la consulta del usuario
                    y seleccionar la herramienta más adecuada de la siguiente lista para responderla.

                    Lista de herramientas disponibles en formato JSON:
                    {tool_specs_json}

                    Basándote en la consulta del usuario y el contexto de la conversación, debes devolver
                    ÚNICAMENTE un objeto JSON con la clave "tool_name" que corresponda al nombre de la
                    herramienta seleccionada.

                    Ejemplo de respuesta:
                    {{"tool_name": "general_conversation"}}

                    No añadas ninguna explicación, comentario o texto adicional fuera del objeto JSON.
                    Tu respuesta debe ser solo el JSON.
                    """

    def dispatch(self, user_prompt: str, history: list, tool_registry: ToolRegistry) -> (str, dict):
        """
        Analiza la consulta del usuario y selecciona la herramienta más apropiada.

        Args:
            user_prompt (str): La consulta actual del usuario.
            history (list): El historial de la conversación.
            tool_registry (ToolRegistry): El registro con todas las herramientas disponibles.

        Returns:
            tuple[str, dict]: Una tupla que contiene:
                - El nombre de la herramienta seleccionada.
                - Un diccionario de argumentos para la herramienta (actualmente vacío).
        """
        import re # Importamos la librería de expresiones regulares

        system_prompt = self._build_system_prompt(tool_registry)

        dispatch_history = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Consulta del usuario: '{user_prompt}'"}
        ]
        
        llm_response_str = self._api_client.generate_content(
            prompt="Selecciona la herramienta apropiada.",
            history=dispatch_history
        )

        # --- INICIO DE LA MEJORA ---
        # Algunos LLMs (como Gemini) pueden devolver el JSON dentro de un bloque de código Markdown.
        # Usamos una expresión regular para extraer el contenido JSON de forma robusta.
        json_match = re.search(r"\{.*\}", llm_response_str, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)
        else:
            clean_json_str = llm_response_str
        # --- FIN DE LA MEJORA ---

        try:
            # Usamos la cadena JSON limpia para el parseo
            response_json = json.loads(clean_json_str)
            tool_name = response_json.get("tool_name")

            if tool_name and tool_registry.get_tool(tool_name):
                print(f"Dispatcher ha seleccionado la herramienta: '{tool_name}'")
                return tool_name, {}
            else:
                print(f"[ADVERTENCIA] El LLM sugirió una herramienta inexistente: '{tool_name}'. Usando la herramienta por defecto.")
        
        except (json.JSONDecodeError, AttributeError):
            print(f"[ADVERTENCIA] La respuesta del LLM no fue un JSON válido: '{llm_response_str}'. Usando la herramienta por defecto.")

        return "general_conversation", {}


