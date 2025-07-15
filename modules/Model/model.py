import torch
import torch.nn as nn
import math

class AudioTransformer(nn.Module):
    def __init__(
            self,
            codebook_count=8,
            vocab_size=1024,
            seq_len=512,
            d_model=512,
            n_heads=8,
            n_layers=6,
            dropout=0.1
    ):
        super().__init__()
        self.codebook_count = codebook_count
        self.d_model = d_model


        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(codebook_count)
        ])

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(codebook_count)
        ])

    def forward(self, x):
        B, L, K = x.shape

        token_emb = sum(
            emb(x[:, :, i]) for i, emb in enumerate(self.embeddings)
        )

        h = token_emb + self.pos_embed[:, :L, :]
        h = self.transformer(h)

        logits = [head(h) for head in self.heads]

        return torch.stack(logits, dim=2)