# model.py
import torch

class TinyTransformer(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads
            ) for _ in range(num_layers)
        ])
        self.fc_out = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x
