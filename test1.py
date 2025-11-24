# test1.py

import torch
import torch.nn.functional as F
import json

from model import TinyTransformer
from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN

# -----------------------------
# Load vocabulary
# -----------------------------

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)  # word -> idx

# keys in vocab are strings, but some may be numeric-like, so ensure int cast
reverse_vocab = {int(idx): word for word, idx in vocab.items()}
VOCAB_SIZE = len(vocab)

# -----------------------------
# Tokenizer / Detokenizer
# (mirror tokenizer.py behavior)
# -----------------------------

def tokenize(text: str):
    # Same logic as tokenizer.tokenize: lowercase + whitespace split
    return [vocab.get(word, 0) for word in text.lower().split()]

def detokenize(tokens):
    words = []
    for tok in tokens:
        if tok == 0:
            continue  # skip PAD
        words.append(reverse_vocab.get(tok, "<unk>"))
    return " ".join(words)

# -----------------------------
# Model setup
# -----------------------------

model = TinyTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
)
model.load_state_dict(torch.load("llms/querycraft_llm.pt", map_location="cpu"))
model.eval()


# -----------------------------
# Generation
# -----------------------------

def generate_sql(full_prompt: str, max_gen_len: int = 80, temperature: float = 0.0):
    """
    Generate SQL for a given prompt.

    For this training setup, the prompt format is:
      "[DB=<db_id>] <natural language question>"

    Example:
      full_prompt = "[DB=department_management] How many heads of the departments are older than 56 ?"
    """

    # Use lowercase forms, because training used text.lower()
    eos_token_id = vocab.get("<eos>")
    sep_token_id = vocab.get("<sep>") or vocab.get("<SEP>")

    # Append <SEP> to mark end of prompt, just like in training:
    # combined_text = prompt + " <SEP> " + response + " <EOS>"
    prompt_text = full_prompt + " <SEP>"

    input_ids = tokenize(prompt_text)
    prompt_len = len(input_ids)

    generated = torch.tensor([input_ids], dtype=torch.long)

    for _ in range(max_gen_len):
        # Respect max sequence length used during training
        input_seq = generated[:, -MAX_SEQ_LEN:]

        with torch.no_grad():
            logits = model(input_seq)

        last_logits = logits[:, -1, :]

        # Sampling vs greedy
        if temperature and temperature > 0.0:
            last_logits = last_logits / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        tok_id = next_token.item()

        # Stop on EOS or SEP token if they exist
        if eos_token_id is not None and tok_id == eos_token_id:
            break
        if sep_token_id is not None and tok_id == sep_token_id:
            break

        generated = torch.cat((generated, next_token), dim=1)

    # Strip the prompt part and detokenize only the generated continuation
    output_tokens = generated[0, prompt_len:].tolist()
    raw_text = detokenize(output_tokens)

    # Optional: trim at first ";" to keep a single SQL statement
    semi_idx = raw_text.find(";")
    if semi_idx != -1:
        raw_text = raw_text[: semi_idx + 1]

    return raw_text.strip()


if __name__ == "__main__":
    # Example matching your training format (no schema, only db_id + question)
    db_id = "department_management"
    question = "How many heads of the departments are older than 56 ?"

    prompt = f"[DB={db_id}] {question}"
    sql_output = generate_sql(prompt, max_gen_len=80, temperature=0.0)

    print("Prompt:\n", prompt)
    print("\nGenerated SQL:\n", sql_output)
