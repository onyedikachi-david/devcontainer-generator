import os
from groq import Groq

def setup_groq():
    api_key = os.environ.get("GROQ_API_KEY")
    return Groq(api_key=api_key)