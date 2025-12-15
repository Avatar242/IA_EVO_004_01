# src/services/gemini_api_client.py

import os
from .base_api_client import BaseApiClient
import google.generativeai as genai

class GeminiApiClient(BaseApiClient):
    """
    Implementación concreta de BaseApiClient para interactuar con la API de Google Gemini.

    Esta clase maneja la configuración de la API, la generación de contenido a través
    de un modelo generativo y la creación de embeddings.
    """

    def __init__(self):
        """
        Inicializa el cliente de Gemini.

        Configura la API de Google y carga los modelos a partir de las variables de entorno.
        Lanza un error si las variables de entorno necesarias no están configuradas.
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.chat_model_name = os.getenv("GEMINI_CHAT_MODEL")
        self.embedding_model_name = os.getenv("GEMINI_EMBEDDING_MODEL")

        if not all([self.api_key, self.chat_model_name, self.embedding_model_name]):
            raise ValueError("Las variables de entorno de Gemini (GOOGLE_API_KEY, GEMINI_CHAT_MODEL, GEMINI_EMBEDDING_MODEL) no están configuradas.")

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.chat_model_name)
            print(f"Cliente Gemini inicializado con los modelos: Chat='{self.chat_model_name}', Embeddings='{self.embedding_model_name}'")
        except Exception as e:
            raise RuntimeError(f"Error al configurar la API de Gemini. Verifica tu API Key. Detalles: {e}")

    def generate_content(self, prompt: str, history: list = None) -> str:
        """
        Genera una respuesta de texto usando el modelo de chat de Gemini.

        Maneja el historial de conversación, convirtiéndolo al formato requerido por la API de Gemini.
        """
        if history is None:
            history = []

        # La API de Gemini requiere un formato específico para el historial.
        # Rol 'model' para las respuestas de la IA.
        # Rol 'user' para las entradas del usuario.
        gemini_history = []
        for message in history:
            role = 'model' if message['role'] == 'assistant' else 'user'
            gemini_history.append({'role': role, 'parts': [message['content']]})
        
        try:
            print(f"\nEnviando a Gemini ({self.chat_model_name}): '{prompt}'")
            chat_session = self.model.start_chat(history=gemini_history)
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            error_message = f"Error al generar contenido con Gemini. Detalles: {e}"
            print(f"\n[ERROR] {error_message}")
            return error_message

    def generate_embeddings(self, text: str) -> list[float]:
        """
        Genera un embedding para un texto dado usando el modelo de embeddings de Gemini.
        """
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT" # Tarea típica para almacenamiento en RAG
            )
            return result['embedding']
        except Exception as e:
            error_message = f"Error al generar embedding con Gemini. Detalles: {e}"
            print(f"\n[ERROR] {error_message}")
            return []