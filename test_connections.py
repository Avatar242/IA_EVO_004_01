# test_connections.py

import os
import sys
from dotenv import load_dotenv

# Añadir el directorio src al path para poder importar módulos de ahí si fuera necesario en el futuro
# Aunque para este script no es estrictamente necesario, es una buena práctica.
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# --- Carga de Variables de Entorno ---
# Carga las variables del archivo .env en el entorno del sistema
load_dotenv()
print("Variables de entorno cargadas.")

# --- Importaciones de Librerías de IA (después de cargar .env) ---
import google.generativeai as genai
import ollama

def test_gemini_connection():
    """
    Prueba la conexión con la API de Google Gemini.
    Configura el cliente, envía un prompt y muestra la respuesta.
    """
    print("\n--- INICIANDO PRUEBA DE CONEXIÓN CON GEMINI ---")
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_CHAT_MODEL")

        if not api_key:
            print("ERROR: La variable de entorno GOOGLE_API_KEY no está configurada.")
            return

        print(f"Usando el modelo: {model_name}")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(model_name=model_name)
        
        prompt = "En una sola frase, ¿por qué el cielo es azul?"
        print(f"Enviando prompt: '{prompt}'")
        
        response = model.generate_content(prompt)
        
        print("\nRespuesta de Gemini:")
        print(response.text)
        print("--- PRUEBA DE GEMINI FINALIZADA CON ÉXITO ---\n")

    except Exception as e:
        print(f"\nERROR: Ocurrió un problema durante la prueba de Gemini.")
        print(f"Detalles del error: {e}")
        print("--- PRUEBA DE GEMINI FALLIDA ---\n")


def test_ollama_connection():
    """
    Prueba la conexión con el servidor local de Ollama.
    Realiza una prueba de chat y una prueba de embeddings.
    """
    print("\n--- INICIANDO PRUEBA DE CONEXIÓN CON OLLAMA ---")
    try:
        chat_model = os.getenv("OLLAMA_CHAT_MODEL")
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")

        # --- Prueba de Chat ---
        print(f"\n[1/2] Probando el modelo de chat: {chat_model}")
        prompt = "En una sola frase, ¿cuál es la capital de Francia?"
        print(f"Enviando prompt: '{prompt}'")
        
        response = ollama.chat(
            model=chat_model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        print("\nRespuesta de Ollama (Chat):")
        print(response['message']['content'])

        # --- Prueba de Embeddings ---
        print(f"\n[2/2] Probando el modelo de embeddings: {embedding_model}")
        embedding_text = "Hola mundo"
        print(f"Generando embedding para el texto: '{embedding_text}'")

        embedding_response = ollama.embeddings(
            model=embedding_model,
            prompt=embedding_text
        )

        print("\nEmbedding generado (primeros 5 valores):")
        # Imprimimos solo una parte para no llenar la consola
        print(embedding_response['embedding'][:5])
        print(f"Dimensiones del embedding: {len(embedding_response['embedding'])}")
        
        print("\n--- PRUEBA DE OLLAMA FINALIZADA CON ÉXITO ---\n")

    except Exception as e:
        print(f"\nERROR: Ocurrió un problema durante la prueba de Ollama.")
        print("Asegúrate de que el servidor de Ollama esté en ejecución.")
        print(f"Detalles del error: {e}")
        print("--- PRUEBA DE OLLAMA FALLIDA ---\n")


if __name__ == "__main__":
    provider = os.getenv("AI_PROVIDER")
    
    print(f"Proveedor de IA seleccionado: {provider}")

    if provider == "gemini":
        test_gemini_connection()
    elif provider == "ollama":
        test_ollama_connection()
    else:
        print(f"ERROR: Proveedor de IA '{provider}' no reconocido.")
        print("Por favor, establece AI_PROVIDER en 'gemini' o 'ollama' en tu archivo .env")