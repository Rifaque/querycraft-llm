# utils.py

from tokenizer import tokenize

MAX_SEQ_LEN = 50  # fallback, can import from config if needed

def prepare_input(text):
    tokens = tokenize(text)
    # pad or truncate
    if len(tokens) < MAX_SEQ_LEN:
        tokens += [0] * (MAX_SEQ_LEN - len(tokens))
    else:
        tokens = tokens[:MAX_SEQ_LEN]
    return tokens
