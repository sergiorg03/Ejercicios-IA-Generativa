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

    # BD con palabras a comparar
    palabras = []

if __name__ == "__main__":
    candidates_filter()