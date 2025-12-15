# Agente de Conocimiento Evolutivo (IA_EVO_004_01)

Este proyecto tiene como objetivo construir un agente de IA avanzado que actúa como una base de conocimiento personal y adaptable, capaz de procesar información multimodal y discernir qué conocimiento persistir a largo plazo.

## Características Principales

- **Ingesta Multimodal:** Procesa archivos, páginas web e imágenes.
- **Discernimiento de Contexto:** Distingue entre memoria a corto y largo plazo (RAG).
- **Base de Conocimientos Actualizable:** Utiliza un sistema RAG para mantener el conocimiento relevante.
- **Motor de IA Dual:** Puede operar con un LLM en la nube (Google Gemini) o un LLM local (vía Ollama).

## Configuración del Entorno

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/Avatar242/IA_EVO_004_01.git
    cd IA_EVO_004_01
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv .venv
    # En Windows
    .\.venv\Scripts\activate
    # En macOS/Linux
    source .venv/bin/activate
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar las variables de entorno:**
    -   Copia el archivo `.env.template` a un nuevo archivo llamado `.env`.
    -   Rellena las variables necesarias en el archivo `.env` (como tu `GOOGLE_API_KEY`).

## Uso

Para iniciar el agente, ejecuta el siguiente comando:
```bash
python src/main.py


---

## Deuda Técnica y Mejoras Futuras

Durante el desarrollo del Hito de RAG, se ha identificado una deuda técnica clave:

*   **Calidad de la Búsqueda (Retrieval Quality):** La combinación actual de la estrategia de chunking (`RecursiveCharacterTextSplitter`) y el modelo de embeddings no es suficientemente precisa para recuperar de forma fiable información específica (como listas o datos concretos) de documentos densos.
    *   **Solución Planificada:** Reemplazar el `RecursiveCharacterTextSplitter` con una implementación de **Chunking Semántico Puro (Nivel 3)**. Esto implicará el uso de modelos de NLP para dividir el texto basándose en cambios de tema, en lugar de separadores de caracteres. Esta tarea se abordará en un hito de optimización futuro.