# src/services/base_api_client.py

from abc import ABC, abstractmethod

class BaseApiClient(ABC):
    """
    Clase base abstracta (Interfaz) para los clientes de API de IA.

    Define un contrato que todas las implementaciones de clientes de API (como Gemini, Ollama, etc.)
    deben seguir. Esto garantiza que el resto de la aplicación pueda interactuar con
    cualquier servicio de IA de una manera uniforme y predecible.

    El uso de 'ABC' (Abstract Base Class) y '@abstractmethod' asegura que si una clase
    hereda de BaseApiClient pero no implementa todos los métodos abstractos,
    se producirá un error en tiempo de instanciación, forzando así el cumplimiento del contrato.
    """

    @abstractmethod
    def generate_content(self, prompt: str, history: list = None) -> str:
        """
        Genera una respuesta de texto a partir de un prompt y un historial de conversación.

        Args:
            prompt (str): La pregunta o instrucción del usuario.
            history (list, optional): Una lista que representa el historial de la
                                      conversación. El formato puede variar según la implementación.
                                      Defaults to None.

        Returns:
            str: La respuesta generada por el modelo de IA.
        """
        pass

    @abstractmethod
    def generate_embeddings(self, text: str) -> list[float]:
        """
        Genera una representación vectorial (embedding) para un texto dado.

        Args:
            text (str): El texto que se convertirá en un embedding.

        Returns:
            list[float]: Una lista de números de punto flotante que representa el
                         embedding del texto.
        """
        pass