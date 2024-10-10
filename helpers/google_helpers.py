import os
from google.generativeai import GenerativeModel

def setup_google():
    api_key = os.environ.get("GOOGLE_API_KEY")
    return GenerativeModel(api_key=api_key)