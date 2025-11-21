# test_mongo.py
import torch
import torch.nn.functional as F
from model import TinyTransformer
from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN
import json

# --- Load Mongo Vocabulary ---
with open("vocab_mongo.json") as f:
    vocab = json.load(f)  # word -> idx

reverse_vocab = {int(idx): word for word, idx in vocab.items()}
VOCAB_SIZE = len(vocab)

# --- Tokenizer / Detokenizer ---

def tokenize(text: str):
    return [vocab.get(word, 0) for word in text.lower().split()]

def detokenize(tokens):
    words = []
    for tok in tokens:
        if tok == 0:
            continue
        words.append(reverse_vocab.get(tok, "<unk>"))
    return " ".join(words)

# --- Load Mongo Transformer Model ---
model = TinyTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
)

model.load_state_dict(torch.load("llms/querycraft_mongo_llm.pt", map_location="cpu"))
model.eval()

# --- Generation Function ---

def generate_mongo(full_prompt: str, max_gen_len: int = 80, temperature: float = 0.0):
    eos_token_id = vocab.get("<EOS>")
    sep_token_id = vocab.get("<SEP>")

    # Training format:   Schema: X\nQuestion: Y <SEP> <TARGET>
    prompt_text = full_prompt + " <SEP>"

    input_ids = tokenize(prompt_text)
    prompt_len = len(input_ids)

    generated = torch.tensor([input_ids], dtype=torch.long)

    for _ in range(max_gen_len):
        input_seq = generated[:, -MAX_SEQ_LEN:]

        with torch.no_grad():
            logits = model(input_seq)

        last_logits = logits[:, -1, :]

        if temperature > 0:
            last_logits = last_logits / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        tok = next_token.item()

        if eos_token_id and tok == eos_token_id:
            break
        if sep_token_id and tok == sep_token_id:
            break

        generated = torch.cat((generated, next_token), dim=1)

    output_tokens = generated[0, prompt_len:].tolist()
    raw = detokenize(output_tokens)

    return raw.strip()

# --- Run Test ---
if __name__ == "__main__":
    schema = "users(_id, name, email, registration_date, country, age, status)"
    question = "Find documents in users where country equals 'IN'"

    prompt = f"Schema: {schema}\nQuestion: {question}"

    result = generate_mongo(prompt, temperature=0.0)

    print("Prompt:\n", prompt)
    print("\nGenerated MongoDB Query:\n", result)
