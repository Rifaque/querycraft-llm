# generate.py
import torch
import torch.nn.functional as F
import json

from model import TinyTransformer
from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN

# --- Load Vocabulary from vocab.json ---
with open("vocab.json") as f:
    vocab = json.load(f)          # word -> idx

reverse_vocab = {idx: word for word, idx in vocab.items()}
VOCAB_SIZE = len(vocab)

def tokenize(text: str):
    return [vocab.get(word, 0) for word in text.lower().split()]

def detokenize(tokens):
    words = []
    for tok in tokens:
        if tok == 0:
            continue
        words.append(reverse_vocab.get(tok, "<unk>"))
    return " ".join(words)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
)
model.load_state_dict(torch.load("llms/querycraft_llm.pt", map_location=device))
model.to(device)
model.eval()

def generate_sql_from_full_prompt(full_prompt: str, max_gen_len: int = 80, temperature: float = 0.0) -> str:
    """
    `full_prompt` should match training format, e.g.:

    Schema: transactions(id, user_id, amount, transaction_date, status)
    Question: Get MAX of amount grouped by user_id from transactions
    """
    eos_token_id = vocab.get("<EOS>")
    sep_token_id = vocab.get("<SEP>")

    # same separator you used during training
    prompt_text = full_prompt + " <SEP>"

    input_ids = tokenize(prompt_text)
    prompt_len = len(input_ids)

    generated = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_gen_len):
        input_seq = generated[:, -MAX_SEQ_LEN:]

        with torch.no_grad():
            logits = model(input_seq)

        last_logits = logits[:, -1, :]

        if temperature and temperature > 0.0:
            last_logits = last_logits / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        tok_id = next_token.item()

        if eos_token_id is not None and tok_id == eos_token_id:
            break
        if sep_token_id is not None and tok_id == sep_token_id:
            break

        generated = torch.cat((generated, next_token), dim=1)

    output_tokens = generated[0, prompt_len:].tolist()
    raw = detokenize(output_tokens)

    # keep only the first SQL statement
    semi_idx = raw.find(";")
    if semi_idx != -1:
        raw = raw[: semi_idx + 1]

    return raw.strip()

def generate_sql(schema: str, question: str, **kwargs) -> str:
    """
    Helper that builds the same prompt structure as your training data.
    """
    full_prompt = f"Schema: {schema}\nQuestion: {question}"
    return generate_sql_from_full_prompt(full_prompt, **kwargs)

if __name__ == "__main__":
    schema = "transactions(id, user_id, amount, transaction_date, status)"
    question = "Get MAX of status grouped by transaction_date from transactions"

    sql = generate_sql(schema, question, max_gen_len=80, temperature=0.0)
    print("Schema:", schema)
    print("Question:", question)
    print("\nGenerated SQL:\n", sql)
