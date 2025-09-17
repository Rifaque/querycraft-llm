# train.py
import torch
import json
from tokenizer import build_vocab, get_vocab_size, tokenize, get_vocab
from model import TinyTransformer
from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN

def pad_sequence(seq, max_len=MAX_SEQ_LEN):
  return seq + [0]*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len]

# Load dataset
with open("data/dataset.jsonl") as f:
  data = [json.loads(line) for line in f]

# Build vocab
build_vocab(data)
VOCAB_SIZE = get_vocab_size()
print("VOCAB_SIZE:", VOCAB_SIZE)

with open("vocab.json", "w") as f:
    json.dump(get_vocab(), f)
print("Updated vocab.json saved successfully.")

model = TinyTransformer(
  vocab_size=VOCAB_SIZE,
  embed_dim=EMBED_DIM,
  num_heads=NUM_HEADS,
  num_layers=NUM_LAYERS,
  max_seq_len=MAX_SEQ_LEN
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # ignore padding

for epoch in range(25): # longer training
  total_loss = 0
  for item in data:
    # combine prompt and response with SEP token
    combined_text = item["prompt"] + " <SEP> " + item["response"]
    input_ids = tokenize(combined_text)
    input_ids = pad_sequence(input_ids)

    input_tensor = torch.tensor([input_ids])
    target_tensor = torch.tensor([input_ids[1:] + [0]]) # shift left, pad end

    outputs = model(input_tensor)
    loss = criterion(outputs.view(-1, VOCAB_SIZE), target_tensor.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "querycraft_llm.pt")
print("Model saved successfully.")