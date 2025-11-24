# tokenizer.py

# Use lowercase special tokens to match .lower()
vocab = {"<pad>": 0, "<sep>": 1, "<eos>": 2}
reverse_vocab = {idx: token for token, idx in vocab.items()}
vocab_index = len(vocab)

def build_vocab(data):
    global vocab, reverse_vocab, vocab_index
    for line in data:
        text = (line["prompt"] + " " + line["response"]).lower().split()
        for word in text:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word
                vocab_index += 1

def tokenize(text):
    return [vocab.get(word, 0) for word in text.lower().split()]

def detokenize(tokens):
    return " ".join([
        reverse_vocab.get(tok, "<unk>")
        for tok in tokens
        if tok != vocab["<pad>"]
    ])

def get_vocab_size():
    return len(vocab)

def get_vocab():
    return vocab
