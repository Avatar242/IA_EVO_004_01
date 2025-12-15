# src/tools/base_tool.py

from abc import ABC, abstractmethod
from typing import Any

class BaseTool(ABC):
    """
    Clase base abstracta (Interfaz) para todas las herramientas del agente.

    Define la estructura y el contrato que cada herramienta debe seguir para ser
    integrada correctamente en el sistema, especialmente con el ToolRegistry y el Dispatcher.

    Una herramienta bien diseñada tiene:
    1. Un 'name' único y programático.
    2. Una 'description' detallada que el LLM utilizará para decidir cuándo usarla.
    3. Un método 'execute' que realiza la acción principal de la herramienta.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Retorna el nombre único y programático de la herramienta.
        Este nombre será utilizado por el Dispatcher para identificar la herramienta a ejecutar.
        Ejemplo: "web_scraper", "rag_search".
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Retorna una descripción clara y concisa de lo que hace la herramienta.
        Esta descripción es crucial, ya que el LLM la analizará para determinar
        si la herramienta es adecuada para responder a la consulta del usuario.
        Debe explicar qué tipo de preguntas puede responder la herramienta.
        """
        pass

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        El punto de entrada principal para la ejecución de la herramienta.

        Este método contiene la lógica principal de la herramienta. Puede aceptar
        cualquier número de argumentos posicionales y de palabra clave, dependiendo
        de la complejidad de la herramienta.

        Args:
            *args (Any): Argumentos posicionales.
            **kwargs (Any): Argumentos de palabra clave.

        Returns:
            Any: El resultado de la ejecución de la herramienta. El tipo de dato
                 del resultado puede variar según la herramienta.
        """
        pass