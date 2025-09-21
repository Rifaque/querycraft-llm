# train-from-checkpoint.py
import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from tokenizer import build_vocab, get_vocab_size, tokenize, get_vocab
from model import TinyTransformer
from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN
from tqdm import tqdm

# --- Directory setup ---
os.makedirs("llm-checkpoints", exist_ok=True)
os.makedirs("llms", exist_ok=True)

# --- Dataset ---
class SQLDataset(Dataset):
    def __init__(self, data_path, max_len=MAX_SEQ_LEN):
        self.data = [json.loads(line) for line in open(data_path)]
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        combined_text = item["prompt"] + " <SEP> " + item["response"] + " <EOS>"
        token_ids = tokenize(combined_text)
        padded_ids = token_ids[:self.max_len] + [0] * (self.max_len - len(token_ids))
        input_tensor = torch.tensor(padded_ids)
        target_tensor = torch.cat((input_tensor[1:], torch.tensor([0])))
        return input_tensor, target_tensor

# --- Vocabulary ---
build_vocab([json.loads(line) for line in open("data/dataset.jsonl")])
VOCAB_SIZE = get_vocab_size()
print("VOCAB_SIZE:", VOCAB_SIZE)
with open("vocab.json", "w") as f:
    json.dump(get_vocab(), f)

# --- DataLoader ---
BATCH_SIZE = 32
train_dataset = SQLDataset("data/dataset.jsonl")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN
)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# --- Load checkpoint ---
checkpoint_path = "llm-checkpoints/querycraft_llm_epoch20.pt"  # <--- set your checkpoint here
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print(f"Resumed from checkpoint → {checkpoint_path}")

# --- Mixed Precision ---
use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler(enabled=use_amp)

# --- Training Loop ---
NUM_EPOCHS = 30  # train additional epochs after checkpoint
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            outputs = model(input_batch)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), target_batch.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        checkpoint_path = f"llm-checkpoints/querycraft_llm_resume_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved → {checkpoint_path}")

# --- Final Model Save ---
torch.save(model.state_dict(), "llms/querycraft_llm_resume.pt")
print("Final resumed model saved → llms/querycraft_llm_resume.pt")
