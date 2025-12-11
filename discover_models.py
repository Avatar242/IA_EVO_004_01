# discover_models.py
import requests
import json

# URL de la API que parece usar la biblioteca de Ollama para listar los modelos
API_URL = "https://ollama.com/api/tags"

def fetch_ollama_models():
    """
    Consulta la API de la biblioteca de Ollama para obtener una lista de modelos disponibles.
    """
    print(f"Consultando la lista de modelos desde {API_URL}...")
    try:
        response = requests.get(API_URL, timeout=15)
        # Lanza un error si la petición no fue exitosa (código de estado no es 2xx)
        response.raise_for_status()
        
        data = response.json()
        
        # --- CORRECCIÓN ---
        # La clave correcta en la respuesta de la API es "models", no "tags".
        models = data.get('models', [])
        
        if not models:
            print("No se encontraron modelos o la respuesta de la API no tuvo el formato esperado.")
            print("Respuesta recibida (primeros 300 caracteres):", response.text[:300])
            return

        print(f"\nSe encontraron {len(models)} modelos en la biblioteca de Ollama:\n")
        
        # Imprimir en un formato de tabla simple
        print(f"{'NOMBRE DEL MODELO':<40} {'TAMAÑO':<15} {'MODIFICADO HACE':<20}")
        print("-" * 75)
        
        # El resto del código asume que la estructura interna de cada modelo es correcta
        for model in sorted(models, key=lambda x: x['name']):
            name = model.get('name', 'N/A')
            # El tamaño viene en bytes, lo convertimos a GB
            size_gb = model.get('size', 0) / (1024**3)
            # Simplificamos la fecha para que solo muestre el día
            modified_at = model.get('modified_at', 'N/A').split('T')[0]

            print(f"{name:<40} {f'{size_gb:.2f} GB':<15} {modified_at:<20}")

    except requests.exceptions.RequestException as e:
        print(f"\nError al conectar con la API de Ollama: {e}")
    except json.JSONDecodeError:
        print("\nError: No se pudo decodificar la respuesta de la API (no es un JSON válido).")
        print("Respuesta recibida:", response.text)

if __name__ == "__main__":
    fetch_ollama_models()