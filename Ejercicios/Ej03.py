import os
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
__API_KEY = os.getenv("GOOGLE_API_KEY")
current_dir = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(current_dir, "Rivas_Guia_basica_uso_inteligencia_artificial_generativa_2025.pdf")
PREGUNTA = "¿Qué es un KPI y pon un ejemplo de la guía?"


def chunk_detective():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(FILE):
        print(FILE)
        print(f"Error, el fichero no existe")
        return

    loader = PyPDFLoader(FILE)
    docs = loader.load()

    # Primera division
    text_splitter_1 = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    chunks_1 = text_splitter_1.split_documents(docs)
    vector_store_1 = FAISS.from_documents(chunks_1, embeddings)

    # Segunda division
    text_splitter_2 = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks_2 = text_splitter_2.split_documents(docs)
    vector_store_2 = FAISS.from_documents(chunks_2, embeddings)

    prompt = ChatPromptTemplate.from_template(
        """
        Utiliza el siguiente contexto para responder a la pregunta del usuario.
        
        Debes SIEMPRE indicar las páginas exactas en la que se encuentra la información.
        Si la pregunta no está literal, intenta deducirla basándote en la información disponible.
        Si el contexto no tiene absolutamente nada que ver, indica qué temas se tratan en el texto.

        Contexto: {context}
        Pregunta: {input}
        """
    )

    document_prompt = ChatPromptTemplate.from_template(
        "Página {metadata[page]}:\n{page_content}"
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)

    rag_chain_1 = create_retrieval_chain(
        vector_store_1.as_retriever(), 
        combine_docs_chain
    )
    rag_chain_2 = create_retrieval_chain(
        vector_store_2.as_retriever(), 
        combine_docs_chain
    )

    response_1 = rag_chain_1.invoke({"input": PREGUNTA})
    response_2 = rag_chain_2.invoke({"input": PREGUNTA})

    """# --- IMPRIMIR EL CONTEXTO ---
    print("\n" + "=" * 50)
    print("CONTEXTO RECUPERADO (Lo que el buscador encontró)")
    print("=" * 50)

    # Recorremos los documentos recuperados
    for i, doc in enumerate(response_1["context"]):
        print(f"\nFRAGMENTO {i + 1} (Página {doc.metadata.get('page', 'N/A')}):")
        print("-" * 30)
        print(doc.page_content)  # Aquí imprimimos el texto real del trozo
        print("-" * 30)"""

    print("\n" + "=" * 50)
    print("RESPUESTA 1 FINAL DE LA IA")
    print("=" * 50)
    print(response_1["answer"])

    print("\n" + "=" * 50)
    print("RESPUESTA 2 FINAL DE LA IA")
    print("=" * 50)
    print(response_2["answer"])


if __name__ == "__main__":
    chunk_detective()
