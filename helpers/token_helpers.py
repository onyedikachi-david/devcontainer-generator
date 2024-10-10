import tiktoken

def count_tokens(text, model="gpt-3.5-turbo-0613"):
    try:
        if model.startswith("claude"):
            # Anthropic models use different tokenization
            return len(text) // 4  # Rough estimate, as Anthropic doesn't provide a public tokenizer
        else:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def truncate_to_token_limit(text, model_name, max_tokens):
    if model_name.startswith("claude"):
        # Rough estimate for Anthropic models
        if len(text) > max_tokens * 4:
            return text[:max_tokens * 4]
        return text
    else:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)
        return text