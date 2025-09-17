# tokenizer.py

vocab = {"<PAD>": 0, "<SEP>": 1}
reverse_vocab = {0: "<PAD>", 1: "<SEP>"}
vocab_index = 2

def build_vocab(data):
    global vocab, reverse_vocab, vocab_index
    for line in data:
        for word in (line["prompt"] + " " + line["response"]).lower().split():
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word
                vocab_index += 1

def tokenize(text):
    return [vocab.get(word, 0) for word in text.lower().split()]

def detokenize(tokens):
    return " ".join([reverse_vocab.get(tok, "<unk>") for tok in tokens if tok != 0])

def get_vocab_size():
    return len(vocab)

def get_vocab():
    """Returns the current vocabulary dictionary."""
    return vocab