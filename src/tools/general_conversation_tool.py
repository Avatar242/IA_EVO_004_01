# src/tools/general_conversation_tool.py

# Importación relativa para la clase base
from .base_tool import BaseTool
# Importación absoluta desde la raíz de 'src' para el cliente de API
from services.base_api_client import BaseApiClient

class GeneralConversationTool(BaseTool):
    """
    Una herramienta para manejar conversaciones generales y preguntas que no
    requieren capacidades específicas de otras herramientas.
    """

    def __init__(self, api_client: BaseApiClient):
        """
        Inicializa la herramienta con un cliente de API para comunicarse con el LLM.

        Args:
            api_client (BaseApiClient): Una instancia de un cliente de API que cumple
                                        con la interfaz BaseApiClient (ej. OllamaApiClient o GeminiApiClient).
        """
        self._api_client = api_client
        print("GeneralConversationTool inicializada.")

    @property
    def name(self) -> str:
        """Retorna el nombre único de la herramienta."""
        return "general_conversation"

    @property
    def description(self) -> str:
        """
        Retorna la descripción que el LLM usará para decidir si usar esta herramienta.
        Es crucial que sea clara y específica.
        """
        return (
            "Útil para mantener una conversación general, saludar, responder preguntas de conocimiento común, "
            "ayudar con tareas creativas como escribir un poema, o para cualquier consulta que no se ajuste "
            "a las capacidades de otras herramientas específicas. Es la herramienta por defecto."
        )

    def execute(self, user_prompt: str, history: list = None) -> str:
        """
        Ejecuta la herramienta pasando el prompt directamente al LLM a través del cliente de API.

        Args:
            user_prompt (str): La entrada del usuario.
            history (list, optional): El historial de la conversación. Defaults to None.

        Returns:
            str: La respuesta generada por el LLM.
        """
        if history is None:
            history = []
            
        return self._api_client.generate_content(prompt=user_prompt, history=history)