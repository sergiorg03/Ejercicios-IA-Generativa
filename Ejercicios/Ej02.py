import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Cargamos la API Key de Gemini
load_dotenv()
__API_KEY = os.getenv("GOOGLE_API_KEY")  

def candidates_filter():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    """ 
        Las opciones con las que se comparará en este caso el puesto ofrecido y obtendrá al mejor candidato mediante la distancia semántica 
        mediante la similitud del coseno (Cosine Similarity) entre vectores euclidianos.
    """
    ejemplos_candidatos = [
        "Experto en Python y desarrollo backend con 5 años de experiencia, especializado en APIs REST, bases de datos (PostgreSQL, MongoDB) y frameworks como FastAPI y Django.", 
        "Desarrollador frontend especializado en React y experiencia de usuario (UX), enfocado en crear interfaces modernas, responsivas y optimizadas para alto rendimiento.",
        "Ingeniero DevOps con experiencia en Docker, CI/CD y cloud (AWS/Azure), encargado de automatizar despliegues, gestionar infraestructura y mejorar la escalabilidad de aplicaciones.",
        "Analista de datos con conocimientos en Python, SQL y visualización (Power BI / Tableau), especializado en transformar datos en insights útiles para la toma de decisiones.",
        "Camarero con experiencia en gestión de sala y atención al cliente, especializado en servir mesas y atender clientes en restaurantes, con gran capacidad de trabajo bajo presión.",
        ]

    vector_db = FAISS.from_texts(ejemplos_candidatos, embeddings)

    print("--- COMPARACIÓN DE DISTANCIAS SEMÁNTICAS ---")
    QUERY_VACANTE = "Buscamos un programador para crear servidores y logica de negocios. "

    resultados = vector_db.similarity_search_with_score(QUERY_VACANTE, k=3)

    print(f"\nVacante: {QUERY_VACANTE}")
    print("\nMejores candidatos:")
    print(resultados)
    
    for i, resultado in resultados:
        print(f"{i}. {resultado.page_content}")

if __name__ == "__main__":
    candidates_filter()