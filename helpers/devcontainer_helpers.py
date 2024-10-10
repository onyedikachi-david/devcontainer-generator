# helpers/devcontainer_helpers.py

import json
import logging
import os
import jsonschema
import tiktoken
from helpers.jinja_helper import process_template
from schemas import DevContainerModel
from supabase_client import supabase
from models import DevContainer
from anthropic_client import Anthropic
from openai_client import OpenAI
from azure_openai_client import AzureOpenAI

import logging
import tiktoken

def truncate_context(context, max_tokens=120000):
    logging.info(f"Starting truncate_context with max_tokens={max_tokens}")
    logging.debug(f"Initial context length: {len(context)} characters")

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoding.encode(context)

    logging.info(f"Initial token count: {len(tokens)}")

    if len(tokens) <= max_tokens:
        logging.info("Context is already within token limit. No truncation needed.")
        return context

    logging.info(f"Context size is {len(tokens)} tokens. Truncation needed.")

    # Prioritize keeping the repository structure and languages
    structure_end = context.find("<<END_SECTION: Repository Structure >>")
    languages_end = context.find("<<END_SECTION: Repository Languages >>")

    logging.debug(f"Structure end position: {structure_end}")
    logging.debug(f"Languages end position: {languages_end}")

    important_content = context[:languages_end] + "<<END_SECTION: Repository Languages >>\n\n"
    remaining_content = context[languages_end + len("<<END_SECTION: Repository Languages >>\n\n"):]

    important_tokens = encoding.encode(important_content)
    logging.debug(f"Important content token count: {len(important_tokens)}")

    if len(important_tokens) > max_tokens:
        logging.warning("Important content alone exceeds max_tokens. Truncating important content.")
        important_content = encoding.decode(important_tokens[:max_tokens])
        return important_content

    remaining_tokens = max_tokens - len(important_tokens)
    logging.info(f"Tokens available for remaining content: {remaining_tokens}")

    truncated_remaining = encoding.decode(encoding.encode(remaining_content)[:remaining_tokens])

    final_context = important_content + truncated_remaining
    final_tokens = encoding.encode(final_context)

    logging.info(f"Final token count: {len(final_tokens)}")
    logging.debug(f"Final context length: {len(final_context)} characters")

    return final_context

def generate_devcontainer_json(client, repo_url, repo_context, devcontainer_url, regenerate=False):
    existing_devcontainer = None
    if "<<EXISTING_DEVCONTAINER>>" in repo_context:
        logging.info("Existing devcontainer.json found in the repository.")
        existing_devcontainer = (
            repo_context.split("<<EXISTING_DEVCONTAINER>>")[1]
            .split("<<END_EXISTING_DEVCONTAINER>>")[0]
            .strip()
        )
        if not regenerate and devcontainer_url:
            logging.info(f"Using existing devcontainer.json from URL: {devcontainer_url}")
            return existing_devcontainer, devcontainer_url

    logging.info("Generating devcontainer.json...")

    # Truncate the context to fit within token limits
    truncated_context = truncate_context(repo_context, max_tokens=126000)

    template_data = {
        "repo_url": repo_url,
        "repo_context": truncated_context,
        "existing_devcontainer": existing_devcontainer
    }

    prompt = process_template("prompts/devcontainer.jinja", template_data)

    if isinstance(client, Anthropic):
        response = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL"),
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        devcontainer_json = response.content
    elif isinstance(client, OpenAI) or isinstance(client, AzureOpenAI):
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        devcontainer_json = response.choices[0].message.content
    elif isinstance(client, GenerativeModel):  # Google's GenerativeModel
        # Placeholder for Google's implementation
        response = client.generate_content(
            prompt=prompt,
            model_name=os.getenv("GOOGLE_MODEL"),
            max_output_tokens=1000
        )
        devcontainer_json = response.text
    elif isinstance(client, Groq):  # Groq's client
        # Placeholder for Groq's implementation
        response = client.generate_content(
            prompt=prompt,
            model_name=os.getenv("GROQ_MODEL"),
            max_output_tokens=1000
        )
        devcontainer_json = response.text
    else:
        raise ValueError(f"Unsupported client type: {type(client)}")

    if validate_devcontainer_json(devcontainer_json):
        logging.info("Successfully generated and validated devcontainer.json")
        if existing_devcontainer and not regenerate:
            return existing_devcontainer, devcontainer_url
        else:
            return devcontainer_json, None  # Return None as URL for generated content
    else:
        logging.warning(f"Generated JSON failed validation")
        raise ValueError("Generated JSON failed validation")

def validate_devcontainer_json(devcontainer_json):
    logging.info("Validating devcontainer.json...")
    schema_path = os.path.join(os.path.dirname(__file__), "..", "schemas", "devContainer.base.schema.json")
    with open(schema_path, "r") as schema_file:
        schema = json.load(schema_file)
    try:
        logging.debug("Running validation...")
        jsonschema.validate(instance=json.loads(devcontainer_json), schema=schema)
        logging.info("Validation successful.")
        return True
    except jsonschema.exceptions.ValidationError as e:
        logging.error(f"Validation failed: {e}")
        return False

def save_devcontainer(new_devcontainer):
    try:
        result = supabase.table("devcontainers").insert(new_devcontainer.dict()).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logging.error(f"Error saving devcontainer to Supabase: {str(e)}")
        raise