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
        tool_specs_json = json.dumps(tool_specs, indent=2)

        return f"""
        Tu rol es ser un despachador inteligente. Tu objetivo es analizar la consulta del usuario
        y seleccionar la herramienta más adecuada de la lista.

        Lista de herramientas disponibles:
        {tool_specs_json}

        Debes devolver ÚNICAMENTE un objeto JSON con dos claves:
        1. "tool_name": El nombre de la herramienta seleccionada.
        2. "tool_args": Un diccionario con los argumentos que la herramienta necesita, extraídos de la consulta del usuario.

        Ejemplo 1: si el usuario dice "hola", la respuesta debe ser:
        {{"tool_name": "general_conversation", "tool_args": {{"user_prompt": "hola"}}}}

        Ejemplo 2: si el usuario pregunta "¿qué dice el documento sobre X?", la respuesta debe ser:
        {{"tool_name": "rag_tool", "tool_args": {{"mode": "query", "user_query": "¿qué dice el documento sobre X?"}}}}

        No añadas explicaciones. Tu respuesta debe ser solo el JSON.
        """

    def dispatch(self, user_prompt: str, history: list, tool_registry: ToolRegistry) -> (str, dict):
        import re
        system_prompt = self._build_system_prompt(tool_registry)

        dispatch_history = [{'role': 'system', 'content': system_prompt}]
        
        # Para el dispatcher, solo pasamos el prompt actual, no el historial completo.
        final_prompt = f"Consulta del usuario: '{user_prompt}'"

        llm_response_str = self._api_client.generate_content(
            prompt=final_prompt,
            history=dispatch_history
        )

        json_match = re.search(r"\{.*\}", llm_response_str, re.DOTALL)
        clean_json_str = json_match.group(0) if json_match else llm_response_str

        try:
            response_json = json.loads(clean_json_str)
            tool_name = response_json.get("tool_name")
            tool_args = response_json.get("tool_args", {}) # Obtenemos los argumentos

            if tool_name and tool_registry.get_tool(tool_name):
                print(f"Dispatcher ha seleccionado la herramienta: '{tool_name}' con args: {tool_args}")
                return tool_name, tool_args # Devolvemos los argumentos
            else:
                print(f"[ADVERTENCIA] El LLM sugirió una herramienta inexistente: '{tool_name}'.")
        
        except (json.JSONDecodeError, AttributeError):
            print(f"[ADVERTENCIA] La respuesta del LLM no fue un JSON válido: '{llm_response_str}'.")

        # Plan B: si todo falla, usar conversación general.
        return "general_conversation", {"user_prompt": user_prompt, "history": history}




