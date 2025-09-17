# test1.py (Corrected)
import torch
import torch.nn.functional as F
from model import TinyTransformer
from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN
import json

# --- Vocabulary Reconstruction ---
with open("vocab.json") as f:
    data = json.load(f)
word_to_idx = data["vocab"]

vocab = {'<PAD>': 0, '<SEP>': 1}
# Note: Add <EOS> here if you plan to retrain with it
# vocab = {'<PAD>': 0, '<SEP>': 1, '<EOS>': 2} 
for word in word_to_idx:
    if word not in vocab:
        vocab[word] = len(vocab)

reverse_vocab = {idx: word for word, idx in vocab.items()}
VOCAB_SIZE = len(vocab)

# --- Tokenizer/Detokenizer ---
def tokenize(text):
    return [vocab.get(word, 0) for word in text.lower().split()]

def detokenize(tokens):
    return " ".join([reverse_vocab.get(tok, "<unk>") for tok in tokens if tok != 0])

# --- Model Initialization ---
model = TinyTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN
)
model.load_state_dict(torch.load("querycraft_llm.pt"))
model.eval()

# --- Generation Function ---
def generate_sql(prompt, max_gen_len=50, temperature=0.7):
    """
    Generates SQL using temperature sampling.
    """
    # Define special token IDs *inside* the function
    eos_token_id = vocab.get("<EOS>") # Will be None if not in vocab
    
    prompt_with_sep = prompt + " <SEP>"
    input_ids = tokenize(prompt_with_sep)
    prompt_len = len(input_ids)
    
    generated = torch.tensor([input_ids])

    for _ in range(max_gen_len):
        input_seq = generated[:, -MAX_SEQ_LEN:]
        
        with torch.no_grad():
            logits = model(input_seq)
        
        last_logits = logits[:, -1, :] / temperature
        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
            
        generated = torch.cat((generated, next_token), dim=1)

    output_tokens = generated[0, prompt_len:].tolist()
    return detokenize(output_tokens)

# --- Example Usage ---
if __name__ == "__main__":
    prompt = "Select all users who"
    sql_output = generate_sql(prompt)
    print("Prompt:", prompt)
    print("Generated SQL:", sql_output)