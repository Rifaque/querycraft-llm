# eval.py

import json
import torch
import torch.nn.functional as F

from model import TinyTransformer
from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN

DATA_PATH = "data/train_spider.json"
TABLES_PATH = "data/tables.json"
MODEL_PATH = "llms/querycraft_llm.pt"
VOCAB_PATH = "vocab.json"

TARGET_DB = "department_management"
MAX_EXAMPLES = 20  # how many examples to eval


# ---------- schema loader (same as train/test) ----------

def load_tables(path):
    with open(path, "r", encoding="utf-8") as f:
        tables = json.load(f)

    by_db = {}
    for t in tables:
        db_id = t["db_id"]
        table_names = t["table_names_original"]
        columns = t["column_names_original"]
        column_types = t["column_types"]

        cols_by_table = {i: [] for i in range(len(table_names))}
        for (tbl_id, col_name), col_type in zip(columns, column_types):
            if tbl_id == -1:
                continue
            cols_by_table[tbl_id].append(f"{col_name}:{col_type}")

        lines = []
        for i, tname in enumerate(table_names):
            cols = cols_by_table[i]
            if cols:
                lines.append(f"{tname}(" + ", ".join(cols) + ")")
            else:
                lines.append(f"{tname}()")

        by_db[db_id] = "\n".join(lines)

    return by_db


SCHEMAS_BY_DB = load_tables(TABLES_PATH)


# ---------- vocab / tokenizer ----------

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
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


# ---------- model ----------

model = TinyTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# ---------- generation ----------

def generate_sql_from_parts(db_id: str, question: str, schema: str,
                            max_gen_len: int = 80, temperature: float = 0.0):
    prompt = f"[DB={db_id}]\nSchema:\n{schema}\nQuestion: {question}"

    eos_token_id = vocab.get("<eos>")
    sep_token_id = vocab.get("<sep>")

    prompt_text = prompt + " <sep>"
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
            probs = torch.softmax(last_logits, dim=-1)
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


# ---------- eval loop ----------

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    # Filter by target DB
    examples = [ex for ex in data if ex.get("db_id") == TARGET_DB]
    examples = examples[:MAX_EXAMPLES]

    schema = SCHEMAS_BY_DB[TARGET_DB]

    correct = 0
    total = len(examples)

    for i, ex in enumerate(examples, 1):
        question = ex["question"]
        gold_sql = ex["query"].strip()

        pred_sql = generate_sql_from_parts(TARGET_DB, question, schema)

        # simple string exact match (you can improve later)
        is_correct = (pred_sql.lower() == gold_sql.lower())
        if is_correct:
            correct += 1

        print(f"\n=== Example {i} ===")
        print("Q:", question)
        print("GOLD:", gold_sql)
        print("PRED:", pred_sql)
        print("CORRECT:", is_correct)

    acc = correct / total if total > 0 else 0.0
    print(f"\nExact match accuracy on {TARGET_DB} ({total} examples): {acc:.3f}")


if __name__ == "__main__":
    main()
