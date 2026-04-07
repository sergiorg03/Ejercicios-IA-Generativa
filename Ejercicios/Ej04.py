# Importamos las librerias necesarias
import os
import re
import sys
from dotenv import load_dotenv
from datetime import datetime, date
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

__API_KEY = os.getenv("GOOGLE_API_KEY")
current_dir = os.path.dirname(os.path.abspath(__file__))
# FILE = os.path.join(current_dir, "")

# Cache en memoria del texto indexado (para respuestas deterministas de "horas/duración")
NORMATIVA_CHUNKS = []

# Fuente única de verdad del calendario
EXAMENES = {
    "Examen Programación Orientada a Objetos": "15 de abril de 2026",
    "Examen Sistemas de Gestión Empresarial": "27 de abril de 2026",
    "Examen Programación Orientada a Objetos": "28 de mayo de 2026",
    "Examen de Lenguajes de Marcas": "7 de abril de 2026",
    "Examen de Bases de datos PL/SQL": "15 de abril de 2026",
}

@tool("calendario_examenes", description="Consulta esta herramienta para buscar información sobre los próximos exámenes, entregas de trabajos y exposiciones.")
def consultar_calendario_examenes():
    docs = [
        Document(
            page_content=f"Evento: {evento}. Fecha: {fecha}",
            metadata={"tipo": "calendario"}
        )
        for evento, fecha in EXAMENES.items()
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    # Guardamos en caché para poder buscar "horas" sin depender del LLM
    global NORMATIVA_CHUNKS
    NORMATIVA_CHUNKS = chunks

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    sys.stdout.write("¡Listo! \n")

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    tool = create_retriever_tool(
        retriever=retriever,
        name="buscador_normativa",
        description="Consulta para buscar información oficial sobre el ciclo, módulos y horas."
    )

    #vector_db_cal = FAISS.from_documents(consultar_calendario_examenes(), embeddings)
    #retriever_cal = vector_db_cal.as_retriever(search_kwargs={"k": 3})    

    #tools_calendario = create_retriever_tool(
    #    retriever=retriever_cal,
    #    name="calendario_examenes",
    #    description="Consulta esta herramienta para buscar información sobre los próximos exámenes, entregas de trabajos y exposiciones."
    #)

    tools = [tool, consultar_calendario_examenes]
    #tools = [tool, tools_calendario]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  # Usamos la versión estable
        temperature=0,
        max_output_tokens=700,  # <-- (500-1000 es ideal para RAG)
        max_retries=2,
    )
    #        - Prioriza el uso de herramientas cuando la pregunta requiera datos exactos como horas, fechas, módulos, normativa, o exámenes.

    system_msg = (
        """
        Eres un asistente educativo amable, conversacional y especializado en ayudar con el Ciclo Formativo.

        Fecha actual: {fecha_actual}

        Dispones de las siguientes herramientas:
        - 'buscador_normativa': para consultas sobre los módulos impartidos y su duración total, contenidos, horas y normativa oficial.
        - 'calendario_examenes': para fechas de exámenes, entregas, exposiciones y eventos académicos.

        REGLAS IMPORTANTES:
        - NO inventes información académica.
        - Las fechas y datos del curso deben ser exactos y obtenidos desde las herramientas.
        - Si no encuentras la respuesta, responde: "No he encontrado esa información en el sistema".
        - La expresión "próximo examen" hace referencia al examen con la fecha más cercana a la fecha actual, sin importar la materia.
        - Cuando el usuario pregunte por el "próximo examen" (o "siguiente examen"), DEBES usar la herramienta 'calendario_examenes',
          comparar las fechas con {fecha_actual}, y responder con el examen más cercano.
        - NO pidas que el usuario especifique la materia si la pregunta es "¿cuál es el próximo examen?".
        - Usa la fecha actual proporcionada como referencia temporal para calcular eventos próximos.

        MUY IMPORTANTE:
        - Debes tener en cuenta el historial de la conversación para responder con cohesión.
        - Si el usuario hace preguntas relacionadas con la anterior, debes inferir el contexto usando el historial.
        - No obligues al usuario a repetir toda la pregunta.
        - Si la pregunta NO está relacionada con el ciclo formativo, módulos, normativa o exámenes, 
            NO utilices herramientas y responde usando tu propio conocimiento de forma natural.
        - La frase "No he encontrado esa información en el sistema" SOLO debe usarse para preguntas académicas.
        - Interpreta preguntas equivalentes como:
            "cuánto dura", "cuántas horas tiene", "carga horaria", "duración del módulo (nombre del módulo)"
        - Si la información no aparece literal, busca sinónimos o formulaciones equivalentes dentro de la normativa antes de responder que no se ha encontrado.

        Ejemplos:
        - Usuario: "¿Cuándo es el examen de BD?"
        - Usuario: "¿Y el de POO?"
        → Debes entender que sigue preguntando por exámenes.

        - Usuario: "¿Cuál es el próximo examen?"
        → Responde con el examen más cercano usando 'calendario_examenes' y {fecha_actual}.

        - Usuario: "¿Cuánto dura Lenguajes de Marcas?"
        - Usuario: "¿Y Bases de Datos?"
        → Debes entender que pregunta por la duración del módulo.

        También puedes responder preguntas generales usando tu propio conocimiento.

        Responde siempre de forma clara, natural y útil.
        Y si la información SE ENCUENTRA en la normativa, responde con la información exacta de la normativa indicando la PÁGINA en la que se encuentra la información.
        """
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
        verbose=True,
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

def _es_pregunta_duracion(texto: str) -> bool:
    t = (texto or "").lower()
    return any(k in t for k in ("cuánto dura", "cuanto dura", "duración", "duracion", "cuántas horas", "cuantas horas", "horas", "carga horaria"))

def _extraer_nombre_modulo(texto: str) -> str | None:
    """
    Intenta extraer el nombre del módulo desde una pregunta tipo:
    - "dime la duración del módulo de Bases de Datos"
    - "¿cuántas horas tiene Bases de Datos?"
    """
    t = (texto or "").strip()
    t_clean = t.strip("¿?").strip()

    m = re.search(r"m[oó]dulo\s+de\s+(.+)$", t_clean, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"(?:horas|duraci[oó]n)\s+(?:tiene|de)\s+(.+)$", t_clean, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # fallback: si es muy corto, asumimos que es el nombre del módulo
    if len(t_clean.split()) <= 6:
        return t_clean

    return None

def _buscar_horas_modulo_en_chunks(modulo: str, chunks) -> tuple[int | None, int | None, str | None]:
    """
    Devuelve (horas, pagina, fragmento) si encuentra un patrón como "192 horas" cerca del nombre del módulo.
    """
    if not modulo:
        return None, None, None

    mod = " ".join(modulo.lower().split())
    horas_re = re.compile(r"(\d{2,4})\s*(?:h|horas)\b", flags=re.IGNORECASE)

    mejores = []
    for doc in chunks or []:
        text = doc.page_content or ""
        t = " ".join(text.lower().split())
        if mod not in t:
            continue

        for mm in horas_re.finditer(text):
            horas = int(mm.group(1))
            pagina = doc.metadata.get("page", None)
            mejores.append((horas, pagina, text))

    if not mejores:
        # Segunda pasada: buscar "horas" aunque el nombre venga separado (p.ej. en tablas)
        palabras = [p for p in re.split(r"\s+", mod) if len(p) > 2]
        for doc in chunks or []:
            text = doc.page_content or ""
            t = " ".join(text.lower().split())
            if not all(p in t for p in palabras):
                continue
            for mm in horas_re.finditer(text):
                horas = int(mm.group(1))
                pagina = doc.metadata.get("page", None)
                mejores.append((horas, pagina, text))

    if not mejores:
        return None, None, None

    # Preferimos valores típicos de módulos (50-400h) y el primer match más razonable
    mejores.sort(key=lambda x: (0 if 50 <= x[0] <= 400 else 1, x[1] if x[1] is not None else 10**9))
    return mejores[0][0], mejores[0][1], mejores[0][2]

def chat_asistente():
    asistente = configurar_asistente()
    if not asistente: return

    fecha_actual = datetime.now().strftime("%d de %B de %Y")
    print(fecha_actual)

    print("\n" + "=" * 40)
    print("SISTEMA DE CONSULTA EDUCATIVA v2.5")
    print("   Escribe 'salir' para finalizar")
    print("=" * 40 + "\n")

    config = {"configurable": {"session_id": "sesion_docente"}}

    while True:
        usuario = input("Tú: ")
        if usuario.lower() in ["salir", "exit"]: break

        try:
            # Respuesta determinista para "duración/horas" (evita alucinaciones y también errores por cuota 429)
            if _es_pregunta_duracion(usuario):
                modulo = _extraer_nombre_modulo(usuario)
                if modulo:
                    horas, pagina, _fragmento = _buscar_horas_modulo_en_chunks(modulo, NORMATIVA_CHUNKS)
                    if horas is not None:
                        if pagina is not None:
                            print(f"Asistente: El módulo {modulo} tiene una duración de {horas} horas (página {pagina + 1}).\n")
                        else:
                            print(f"Asistente: El módulo {modulo} tiene una duración de {horas} horas.\n")
                        continue

            # Invocamos al agente
            response = asistente.invoke({"input": usuario, "fecha_actual": fecha_actual}, config=config)

            # PASO CRÍTICO: Limpiamos la respuesta antes de mostrarla
            respuesta_final = limpiar_respuesta(response["output"])

            print(f"Asistente: {respuesta_final}\n")

        except Exception as e:
            print(f"Error en la comunicación: {e}")

if __name__ == "__main__":
    chat_asistente()