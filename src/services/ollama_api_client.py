# src/services/ollama_api_client.py

import os
from .base_api_client import BaseApiClient
import ollama

class OllamaApiClient(BaseApiClient):
    """
    Implementación concreta de BaseApiClient para interactuar con un servidor local de Ollama.

    Esta clase maneja la comunicación con los endpoints de chat y embeddings de Ollama,
    utilizando los modelos especificados en las variables de entorno.
    """

    def __init__(self):
        """
        Inicializa el cliente de Ollama.
        
        Lee los nombres de los modelos de chat y embeddings desde las variables de entorno.
        Lanza un error si las variables de entorno necesarias no están configuradas.
        """
        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
        
        if not self.chat_model or not self.embedding_model:
            raise ValueError("Las variables de entorno de Ollama (OLLAMA_CHAT_MODEL, OLLAMA_EMBEDDING_MODEL) no están configuradas.")
        
        print(f"Cliente Ollama inicializado con los modelos: Chat='{self.chat_model}', Embeddings='{self.embedding_model}'")

    def generate_content(self, prompt: str, history: list = None) -> str:
        """
        Genera una respuesta de texto usando el modelo de chat de Ollama.

        Args:
            prompt (str): La pregunta o instrucción del usuario.
            history (list, optional): El historial de la conversación. 
                                      Se espera una lista de diccionarios con claves 'role' y 'content'.
                                      Defaults to None.

        Returns:
            str: La respuesta generada por el modelo.
        """
        if history is None:
            history = []

        # Añadimos el prompt actual del usuario al historial para la llamada a la API
        messages = history + [{'role': 'user', 'content': prompt}]

        try:
            print(f"\nEnviando a Ollama ({self.chat_model}): '{prompt}'")
            response = ollama.chat(
                model=self.chat_model,
                messages=messages
            )
            return response['message']['content']
        
        except Exception as e:
            # Capturamos una excepción genérica, que suele ser por problemas de conexión.
            error_message = f"Error al conectar con el servidor de Ollama. Asegúrate de que está en ejecución. Detalles: {e}"
            print(f"\n[ERROR] {error_message}")
            return error_message

    def generate_embeddings(self, text: str) -> list[float]:
        """
        Genera un embedding para un texto dado usando el modelo de embeddings de Ollama.

        Args:
            text (str): El texto a convertir en embedding.

        Returns:
            list[float]: El embedding generado. Retorna una lista vacía en caso de error.
        """
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response['embedding']
        
        except Exception as e:
            error_message = f"Error al generar embedding con Ollama. Asegúrate de que el servidor está en ejecución. Detalles: {e}"
            print(f"\n[ERROR] {error_message}")
            return []