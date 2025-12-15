# src/core/tool_registry.py

from typing import List, Dict, Type
# Corregimos la ruta de importación para que sea absoluta desde 'src'
from tools.base_tool import BaseTool

class ToolRegistry:
    """
    Un catálogo centralizado para registrar y gestionar todas las herramientas del agente.

    Esta clase mantiene un registro de todas las instancias de herramientas disponibles,
    permitiendo al Dispatcher acceder a ellas y a sus especificaciones
    (nombre y descripción) de una manera estructurada.
    """

    def __init__(self):
        """
        Inicializa el ToolRegistry con un diccionario vacío para almacenar las herramientas.
        """
        self._tools: Dict[str, BaseTool] = {}
        print("ToolRegistry inicializado.")

    def register_tool(self, tool_instance: BaseTool):
        """
        Registra una nueva instancia de herramienta en el catálogo.

        La herramienta debe ser una instancia de una clase que herede de BaseTool.
        El nombre de la herramienta se utiliza como clave para un acceso rápido.

        Args:
            tool_instance (BaseTool): La instancia de la herramienta a registrar.

        Raises:
            TypeError: Si el objeto proporcionado no es una instancia de BaseTool.
            ValueError: Si ya existe una herramienta registrada con el mismo nombre.
        """
        if not isinstance(tool_instance, BaseTool):
            raise TypeError("La herramienta a registrar debe ser una instancia de una clase que herede de BaseTool.")
        
        tool_name = tool_instance.name
        if tool_name in self._tools:
            raise ValueError(f"Ya existe una herramienta registrada con el nombre '{tool_name}'.")
        
        self._tools[tool_name] = tool_instance
        print(f"Herramienta '{tool_name}' registrada exitosamente.")

    def get_tool(self, tool_name: str) -> BaseTool:
        """
        Recupera una herramienta del registro por su nombre.

        Args:
            tool_name (str): El nombre de la herramienta a recuperar.

        Returns:
            BaseTool: La instancia de la herramienta si se encuentra.

        Raises:
            KeyError: Si no se encuentra ninguna herramienta con el nombre proporcionado.
        """
        if tool_name not in self._tools:
            raise KeyError(f"No se encontró ninguna herramienta con el nombre '{tool_name}'.")
        return self._tools[tool_name]

    def get_tool_specifications(self) -> List[Dict[str, str]]:
        """
        Genera una lista de especificaciones para todas las herramientas registradas.

        Esta lista está diseñada para ser fácilmente convertible a JSON y ser utilizada
        por el LLM en el prompt del Dispatcher para la selección de herramientas.

        Returns:
            List[Dict[str, str]]: Una lista de diccionarios, donde cada diccionario
                                 contiene el 'name' y la 'description' de una herramienta.
        """
        return [
            {"name": name, "description": tool.description}
            for name, tool in self._tools.items()
        ]