import os
from anthropic import Anthropic

def setup_anthropic():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    client = Anthropic(api_key=api_key)
    return client, model