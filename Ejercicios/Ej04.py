# Importamos las librerias necesarias
import os
import sys
from dotenv import load_dotenv
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

__API_KEY = os.getenv("GOOGLE_API_KEY")
current_dir = os.path.dirname(os.path.abspath(__file__))
# FILE = os.path.join(current_dir, "")

def consultar_calendario_examenes():
    data = {
        "Entrega trabajo POO": "7 de abril de 2026",
        "Entrega boletin ADP" : "27 de abril de 2026",
        "Speaking C1" : "28 de mayo de 2026",
        "Exposicion sobre la IA" : "7 de abril de 2026",
    }

    docs = [
        Document(
            page_content=f"{evento}: {fecha}",
            metadata={"tipo": "calendario"}
        )
        for evento, fecha in data.items()
    ]
    return docs

def configurar_asistente():
    if not os.path.exists(os.path.join(current_dir, "normativa")):
        os.makedirs(os.path.join(current_dir, "normativa"))
        print("Añade tus PDFs dentro de la carpeta normativa.")
        return None

    sys.stdout.write("--- Indexando normativa... ")
    sys.stdout.flush()

    loader = PyPDFDirectoryLoader(os.path.join(current_dir, "normativa/"))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    sys.stdout.write("¡Listo! \n")

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    tool = create_retriever_tool(
        retriever=retriever,
        name="buscador_normativa",
        description="Consulta para buscar información oficial sobre el ciclo, módulos y horas."
    )

    #embeddings_cal = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db_cal = FAISS.from_documents(consultar_calendario_examenes(), embeddings)
    retriever_cal = vector_db_cal.as_retriever(search_kwargs={"k": 5})    

    tools_calendario = create_retriever_tool(
        retriever=retriever_cal,
        name="calendario_examenes",
        description="Consulta para buscar información sobre los siguiente examenes, entrega de trabajos y exposiciones."
    )

    tools = [tool, tools_calendario]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  # Usamos la versión estable
        temperature=0,
        max_output_tokens=600,  # <-- (500-1000 es ideal para RAG)
        max_retries=2,
    )

    system_msg = (
        """Eres un asistente versátil y amable. Tu especialidad es ayudar con el Ciclo Formativo 
        usando la herramienta 'buscador_normativa' para consultas específicas sobre módulos y horas. 
        Y la herramienta 'calendario_examenes' para consultas sobre fechas de examenes, entregas de trabajos y exposiciones. 
        Sin embargo, si el usuario te pregunta sobre otros temas generales (como cocina, cultura o ayuda general), 
        responde usando tu propio conocimiento de forma cordial."""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True
    )

    # Memoria persistente durante la ejecución
    history = ChatMessageHistory()

    return RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

def limpiar_respuesta(salida_raw):
    """Extrae únicamente el texto de la respuesta de Gemini."""
    if isinstance(salida_raw, list):
        texto = ""
        for item in salida_raw:
            if isinstance(item, dict) and 'text' in item:
                texto += item['text']
            elif isinstance(item, str):
                texto += item
        return texto
    return str(salida_raw)

def chat_asistente():
    asistente = configurar_asistente()
    if not asistente: return

    print("\n" + "=" * 40)
    print("SISTEMA DE CONSULTA EDUCATIVA v2.5")
    print("   Escribe 'salir' para finalizar")
    print("=" * 40 + "\n")

    config = {"configurable": {"session_id": "sesion_docente"}}

    while True:
        usuario = input("Tú: ")
        if usuario.lower() in ["salir", "exit"]: break

        try:
            # Invocamos al agente
            response = asistente.invoke({"input": usuario}, config=config)

            # PASO CRÍTICO: Limpiamos la respuesta antes de mostrarla
            respuesta_final = limpiar_respuesta(response["output"])

            print(f"Asistente: {respuesta_final}\n")

        except Exception as e:
            print(f"Error en la comunicación: {e}")

if __name__ == "__main__":
    chat_asistente()