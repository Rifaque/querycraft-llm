import torch
import torch.nn.functional as F
import json

from model import TinyTransformer
from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN

# --- Load vocab ---
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

reverse_vocab = {int(idx): word for word, idx in vocab.items()}
VOCAB_SIZE = len(vocab)

def tokenize(text: str):
    return [vocab.get(word, 0) for word in text.lower().split()]

def detokenize(tokens):
    words = []
    pad_id = vocab.get("<pad>", 0)
    for tok in tokens:
        if tok == pad_id:
            continue
        words.append(reverse_vocab.get(tok, "<unk>"))
    return " ".join(words)

model = TinyTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
)
model.load_state_dict(torch.load("llms/querycraft_llm.pt", map_location="cpu"))
model.eval()

def generate_sql(full_prompt: str, max_gen_len: int = 80, temperature: float = 0.0):
    eos_token_id = vocab.get("<eos>")
    sep_token_id = vocab.get("<sep>")

    # same as training: prompt + <sep>
    prompt_text = full_prompt + " <sep>"

    input_ids = tokenize(prompt_text)
    prompt_len = len(input_ids)
    generated = torch.tensor([input_ids], dtype=torch.long)

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
    raw_text = detokenize(output_tokens)

    semi_idx = raw_text.find(";")
    if semi_idx != -1:
        raw_text = raw_text[: semi_idx + 1]

    return raw_text.strip()


if __name__ == "__main__":
    db_id = "department_management"
    question = "How many heads of the departments are older than 56 ?"

    # minimal hand-written schema example
    schema = (
        "head(id, name, born_state, age)\n"
        "department(id, name, creation, budget_in_billions)"
    )

    prompt = f"[DB={db_id}]\nSchema:\n{schema}\nQuestion: {question}"

    sql_output = generate_sql(prompt, max_gen_len=80, temperature=0.0)

    print("Prompt:\n", prompt)
    print("\nGenerated SQL:\n", sql_output)
