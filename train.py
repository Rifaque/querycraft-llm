# train.py

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tokenizer import build_vocab, get_vocab_size, tokenize, get_vocab
from model import TinyTransformer
from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN

# -----------------------------
# Paths / Hyperparameters
# -----------------------------
DATA_PATH = "data/train_spider.json"   # <-- put your new JSON file here
BATCH_SIZE = 32
NUM_EPOCHS = 2
LR = 1e-4

os.makedirs("llm-checkpoints", exist_ok=True)
os.makedirs("llms", exist_ok=True)

# -----------------------------
# Data loading + conversion
# -----------------------------


def load_spider_style_data(path):
    """
    Load data which can be either:
      - A JSON array: [ {...}, {...}, ... ]
      - Or JSONL: one JSON object per line.
    Returns: list[dict]
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # standard JSON array
            data = json.load(f)
        else:
            # JSONL
            data = [json.loads(line) for line in f if line.strip()]
    return data


def example_to_prompt_response(ex):
    """
    Convert a Spider-style example into the old {prompt, response} format
    expected by tokenizer.build_vocab().
    """
    db_id = ex.get("db_id", "unknown_db")
    question = ex["question"]
    sql_query = ex["query"]

    # You can later enhance this to include schema context etc.
    prompt = f"[DB={db_id}] {question}"
    response = sql_query

    return {"prompt": prompt, "response": response}


# -----------------------------
# Dataset
# -----------------------------


class SQLDataset(Dataset):
    def __init__(self, data_path, max_len=MAX_SEQ_LEN):
        self.raw_data = load_spider_style_data(data_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        ex = self.raw_data[idx]
        pr = example_to_prompt_response(ex)

        # Combine prompt + response like before
        combined_text = pr["prompt"] + " <sep> " + pr["response"] + " <eos>"
        token_ids = tokenize(combined_text)  # uses your tokenizer

        # Pad / truncate
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        else:
            token_ids = token_ids + [0] * (self.max_len - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)

        # Next-token prediction: shift targets by one
        target_ids = torch.empty_like(input_ids)
        target_ids[:-1] = input_ids[1:]
        target_ids[-1] = 0  # PAD as dummy last target (ignored in loss)

        return input_ids, target_ids


# -----------------------------
# Build vocab from new data
# -----------------------------

print("Loading raw data...")
raw_data = load_spider_style_data(DATA_PATH)

print("Converting to {prompt, response} for vocab building...")
processed_for_vocab = [example_to_prompt_response(ex) for ex in raw_data]

print("Building vocabulary...")
build_vocab(processed_for_vocab)
VOCAB_SIZE = get_vocab_size()
print("VOCAB_SIZE:", VOCAB_SIZE)

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(get_vocab(), f, ensure_ascii=False, indent=2)
print("Updated vocab.json saved successfully.")

# -----------------------------
# DataLoader
# -----------------------------

train_dataset = SQLDataset(DATA_PATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Model / Optimizer / Loss
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

model = TinyTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD token

# Mixed precision (only used if CUDA is available)
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# -----------------------------
# Training loop
# -----------------------------

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_batch)  # (batch, seq_len, vocab_size)
            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                target_batch.view(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg Loss: {avg_loss:.4f}")

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        ckpt_path = f"llm-checkpoints/querycraft_llm_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")

# -----------------------------
# Save final model
# -----------------------------
final_path = "llms/querycraft_llm.pt"
torch.save(model.state_dict(), final_path)
print(f"Final model saved successfully → {final_path}")
