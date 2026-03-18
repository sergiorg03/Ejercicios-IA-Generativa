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

    Eres un experto en informatica encargado en la explicacion de errores tecnicos de programacion.

    Tu tarea es traducir cualquier error que ocurra al ejecutar codigo y explicarlo de forma clara al usuario.

    Debes adaptar SIEMPRE la explicacion al nivel del usuario:
    Si es un niño: usa lenguaje muy simple, frases cortas y ejemplos faciles de entender.
    Si es un estudiante de DAW: usa un lenguaje tecnico moderado y explica el porqué del error.
    Si es un CEO con prisa: sé muy breve, directo y sin tecnicismos innecesarios.

    No mezcles estilos. Adapata completamente la respuesta al usuario.

    Despues de explicar el error, proporciona una solucion en una sola linea de codigo clara y funcional.

    Nivel del usuario: {user_level}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", personalidad), # definimos las instrucciones al modelo
        ("human", "Traduce este error: {error}")
    ])

    # Encadenamos el prompt al modelo
    chain = prompt | llm

    error_sample = "NullPointerException"
    nivel_usuario = "Jefe"

    # Ejecutamos la cadena
    resultado = chain.invoke({"error": error_sample, "user_level": nivel_usuario})

    # Imprimimos el resultado
    print("\nTraduciendo error para el usuario: \n\n")
    print(resultado.content)

# MAIN
if __name__ == "__main__":
    traductor_jerga()