# src/core/dispatcher.py

import json
import re
from services.base_api_client import BaseApiClient
from core.tool_registry import ToolRegistry

class Dispatcher:
    """
    MODIFICADO: Decide qué herramienta usar. Su única responsabilidad es devolver
    el nombre de la herramienta más adecuada.
    """

    def __init__(self, api_client: BaseApiClient):
        self._api_client = api_client
        print("Dispatcher (simplificado) inicializado.")

    def _build_system_prompt(self, tool_registry: ToolRegistry) -> str:
        """
        Construye un prompt más simple que solo pide el nombre de la herramienta.
        """
        tool_specs = tool_registry.get_tool_specifications()
        tool_specs_json = json.dumps(tool_specs, indent=2)

        return f"""
Tu rol es ser un despachador inteligente. Tu objetivo es analizar la consulta del usuario
y seleccionar la herramienta más adecuada de la siguiente lista para responderla.

Lista de herramientas disponibles en formato JSON:
{tool_specs_json}

Basándote en la consulta, responde ÚNICAMENTE con el string del nombre de la herramienta
seleccionada. Por ejemplo: "rag_tool" o "general_conversation".

No añadas ninguna explicación, comentario o formato JSON. Tu respuesta debe ser solo
el nombre de la herramienta.
"""

    def dispatch(self, user_prompt: str, history: list, tool_registry: ToolRegistry) -> str:
        """
        MODIFICADO: Devuelve solo el nombre (str) de la herramienta seleccionada.
        """
        system_prompt = self._build_system_prompt(tool_registry)

        dispatch_history = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Consulta del usuario: '{user_prompt}'"}
        ]
        
        # El LLM solo debe devolver el nombre de la herramienta como un string simple
        tool_name = self._api_client.generate_content(
            prompt="Selecciona la herramienta apropiada.",
            history=dispatch_history
        ).strip().replace("\"", "") # Limpiamos por si devuelve comillas

        try:
            # Verificamos si la herramienta seleccionada realmente existe
            tool_registry.get_tool(tool_name)
            print(f"Dispatcher ha seleccionado la herramienta: '{tool_name}'")
            return tool_name
        except KeyError:
            # Si el LLM alucina un nombre de herramienta que no existe
            print(f"[ADVERTENCIA] El LLM sugirió una herramienta inexistente: '{tool_name}'. Usando la herramienta por defecto.")
            return "general_conversation"