import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Cargamos la API Key de Gemini
load_dotenv()
__API_KEY = os.getenv("GOOGLE_API_KEY")

def traductor_jerga():

    # Iniciamos el LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=__API_KEY,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Personalidad y tarea (Son las instrucciones al modelo)
    personalidad = """
    Eres un experto en informatica encargado de la traduccion de errores.
    Tu tarea es traducir cualquier error producido a la hora de ejecutar un codigo y explicarsela al usuario.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", personalidad), # definimos las instrucciones al modelo
        ("human", "Traduce este error: {error}")
    ])

    # Encadenamos el prompt al modelo
    chain = prompt | llm

    error_sample = "SyntaxError: invalid syntax"

    # Ejecutamos la cadena
    resultado = chain.invoke({"error": error_sample})

    # Imprimimos el resultado
    print("Traduciendo error para el usuario: ")
    print(resultado.content)



if __name__ == "__main__":
    traductor_jerga()