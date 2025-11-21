#!/usr/bin/env python3
import argparse
import json
import torch
import torch.nn.functional as F
from model import TinyTransformer
from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN


# ----------------------------
# Tokenizer Helpers
# ----------------------------
def load_vocab(path):
    with open(path) as f:
        vocab = json.load(f)
    reverse = {int(i): w for w, i in vocab.items()}
    return vocab, reverse

def tokenize(text, vocab):
    return [vocab.get(w, 0) for w in text.lower().split()]

def detokenize(tokens, reverse_vocab):
    return " ".join(reverse_vocab.get(t, "<unk>") for t in tokens if t != 0)


# ----------------------------
# Model Loader
# ----------------------------
def load_model(vocab_size, weight_path, device):
    model = TinyTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ----------------------------
# Generation
# ----------------------------
def generate(model, vocab, reverse_vocab, full_prompt, max_gen_len=80, temperature=0.0):
    eos_id = vocab.get("<EOS>")
    sep_id = vocab.get("<SEP>")

    prompt_text = full_prompt + " <SEP>"
    input_ids = tokenize(prompt_text, vocab)
    prompt_len = len(input_ids)

    device = next(model.parameters()).device
    generated = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_gen_len):
        x = generated[:, -MAX_SEQ_LEN:]

        with torch.no_grad():
            logits = model(x)

        last_logits = logits[:, -1, :]

        if temperature > 0:
            last_logits = last_logits / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        tok = next_token.item()

        if eos_id and tok == eos_id:
            break
        if sep_id and tok == sep_id:
            break

        generated = torch.cat((generated, next_token), dim=1)

    out_tokens = generated[0, prompt_len:].tolist()
    text = detokenize(out_tokens, reverse_vocab)

    # Trim multiple queries
    semi = text.find(";")
    if semi != -1:
        text = text[: semi + 1]

    return text.strip()


# ----------------------------
# Main CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="QueryCraft Unified CLI")
    parser.add_argument("mode", choices=["sql", "mongo"], help="Which model to use")
    parser.add_argument("schema", help="Schema: table(columns...) or collection(fields...)")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    mode = args.mode.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------
    # Choose model + vocab by mode
    # -----------------------------------------
    if mode == "sql":
        vocab_path = "vocab.json"
        weight_path = "llms/querycraft_llm.pt"
    else:
        vocab_path = "vocab_mongo.json"
        weight_path = "llms/querycraft_mongo_llm.pt"

    vocab, reverse_vocab = load_vocab(vocab_path)
    model = load_model(len(vocab), weight_path, device)

    # Unified prompt format
    full_prompt = f"Schema: {args.schema}\nQuestion: {args.question}"

    output = generate(
        model,
        vocab,
        reverse_vocab,
        full_prompt,
        temperature=args.temp,
    )

    print("\n==================== OUTPUT ====================\n")
    print(output)
    print("\n================================================\n")


if __name__ == "__main__":
    main()
