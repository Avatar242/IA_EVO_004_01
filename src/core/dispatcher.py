# src/core/dispatcher.py
import re
import json
from services.base_api_client import BaseApiClient
from core.tool_registry import ToolRegistry

class Dispatcher:
    """
    MODIFICADO (v3): Usa una estrategia de dos pasos para mayor robustez.
    1. Llama al LLM para elegir solo el NOMBRE de la herramienta.
    2. Si es necesario, hace una segunda llamada para extraer parámetros específicos (como filtros).
    """
    def __init__(self, api_client: BaseApiClient):
        self._api_client = api_client
        print("Dispatcher (estrategia de 2 pasos) inicializado.")

    def _choose_tool(self, user_prompt: str, tool_registry: ToolRegistry) -> str:
        """
        Paso 1: Llama al LLM para que elija la herramienta más adecuada.
        """
        tool_specs = tool_registry.get_tool_specifications()
        tool_specs_json = json.dumps(tool_specs, indent=2)

        system_prompt = f"""
Tu rol es ser un despachador inteligente. Tu objetivo es analizar la consulta del usuario
y seleccionar la herramienta más adecuada de la lista.

Lista de herramientas disponibles:
{tool_specs_json}

Responde ÚNICAMENTE con el string del nombre de la herramienta seleccionada.
Ejemplo: "rag_tool" o "general_conversation".
No añadas explicaciones ni formato.
"""
        dispatch_history = [{'role': 'system', 'content': system_prompt}]
        
        tool_name = self._api_client.generate_content(
            prompt=f"Consulta del usuario: '{user_prompt}'",
            history=dispatch_history
        ).strip().replace("\"", "")

        try:
            tool_registry.get_tool(tool_name)
            print(f"Dispatcher (Paso 1): Herramienta elegida -> '{tool_name}'")
            return tool_name
        except KeyError:
            print(f"[ADVERTENCIA] El LLM sugirió una herramienta inexistente: '{tool_name}'. Usando plan B.")
            return "general_conversation"

    def _extract_filter(self, user_prompt: str) -> dict | None:
        """
        Paso 2: Si se eligió RAG, llama al LLM para extraer una posible categoría de filtro.
        """
        system_prompt = """
Tu rol es ser un asistente de extracción de entidades. Analiza la siguiente consulta del usuario.
Si la consulta menciona una categoría de documento específica (ej: 'seguridad', 'finanzas', 'ciencia'),
extrae esa categoría.

Responde ÚNICAMENTE con un objeto JSON con una clave "category".
Si no se menciona ninguna categoría, responde con un JSON y el valor null.

EJEMPLOS:
-   Usuario: "en el informe de ciberseguridad, qué son las APIs?"
    {{"category": "ciberseguridad"}}
-   Usuario: "qué son las 11 amenazas?"
    {{"category": null}}
"""
        history = [{'role': 'system', 'content': system_prompt}]
        
        response_str = self._api_client.generate_content(
            prompt=f"Consulta del usuario: '{user_prompt}'",
            history=history
        )
        
        try:
            # Reutilizamos la lógica de extracción de JSON robusta
            json_match = re.search(r"\{.*\}", response_str, re.DOTALL)
            clean_json_str = json_match.group(0) if json_match else response_str
            data = json.loads(clean_json_str)

            category = data.get("category")
            if category:
                print(f"Dispatcher (Paso 2): Filtro de categoría extraído -> '{category}'")
                return {"category": category}
            else:
                print("Dispatcher (Paso 2): No se extrajo ningún filtro de categoría.")
                return None
        except (json.JSONDecodeError, AttributeError):
            print("[ADVERTENCIA] No se pudo extraer el filtro. Se procederá sin filtro.")
            return None

    def dispatch(self, user_prompt: str, history: list, tool_registry: ToolRegistry) -> (str, dict):
        # Paso 1: Elegir la herramienta
        tool_name = self._choose_tool(user_prompt, tool_registry)

        # Paso 2: Construir los argumentos en código Python
        tool_args = {}
        if tool_name == "rag_tool":
            # Si la herramienta es RAG, intentamos extraer un filtro
            where_filter = self._extract_filter(user_prompt)
            tool_args = {
                "mode": "query",
                "user_query": user_prompt,
                "where_filter": where_filter
            }
        elif tool_name == "general_conversation":
            tool_args = {
                "user_prompt": user_prompt,
                "history": history
            }
        
        print(f"Dispatcher (Final): Plan de ejecución -> Herramienta='{tool_name}', Args={tool_args}")
        return tool_name, tool_args