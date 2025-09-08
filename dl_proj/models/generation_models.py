import torch
import torch.nn as nn
from my_transformers.layers import ReformerEncoderLayer, EncoderLayer


class MyGenerator(nn.Module):
    def __init__(self, configs):
        super(MyGenerator, self).__init__()
        self.configs = configs

        # MODULES
        self.embedding = nn.Embedding(num_embeddings=configs.vocab_size + 2, embedding_dim=configs.embedding_dim)
        self.pos_embedding = nn.Embedding(configs.max_len, configs.embedding_dim)
        self.layer_norm = nn.LayerNorm(configs.embedding_dim)
        self.dropout = nn.Dropout(configs.dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(
                model_dim=configs.embedding_dim,
                heads=configs.heads,
                d_k=configs.d_k,
                d_v=configs.d_v,
                expansion_factor=configs.expansion_factor,
                attn_dropout=configs.attn_dropout,
                dropout=configs.dropout,
            ) for _ in range(configs.depth)
        ])

        self.out = nn.Sequential(
            nn.Linear(configs.embedding_dim, configs.embedding_dim),
            nn.ReLU(),
            nn.Linear(configs.embedding_dim, configs.vocab_size + 2)
        )

        # Causal Mask
        self.causal_mask = torch.tril(torch.ones(configs.max_len, configs.max_len)).to(configs.device)
        # Positions vector
        self.register_buffer('pos', torch.arange(0, configs.max_len, dtype=torch.long).unsqueeze(0))

    def _embeddings(self, x):
        emb = self.embedding(x)
        pos_emb = self.pos_embedding(self.pos)
        x = emb + pos_emb[:, :x.shape[1], :]
        return x

    def _output(self, x):
        x = self.layer_norm(x)
        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, mask=self.causal_mask if self.configs.causal_mask else None)

        x = self.out(x)
        return x


    def forward(self, x):
        x = self._embeddings(x)
        x = self._output(x)
        return x

class MyGeneratorLSH(MyGenerator):
    def __init__(self, configs):
        super(MyGeneratorLSH, self).__init__(configs)

        self.layers = nn.ModuleList([
            ReformerEncoderLayer(
                model_dim=configs.embedding_dim,
                heads=configs.heads,
                d_k=configs.d_k,
                d_v=configs.d_v,
                expansion_factor=configs.expansion_factor,
                dropout=configs.dropout,
                attn_dropout=configs.attn_dropout,
                device=configs.device,
                n_rounds=configs.n_rounds,
                max_len=configs.max_len,
                bucket_size=configs.bucket_size,
                self_attn_mask=configs.self_attn_mask,
            ) for _ in range(configs.depth)
        ])

        # Causal Mask
        self.causal_mask = torch.tril(torch.ones(2 * configs.bucket_size, 2 * configs.bucket_size)).to(configs.device)

class CondGenerator(MyGenerator):
    def __init__(self, configs, w_noise = 0.05):
        super(CondGenerator, self).__init__(configs)
        self.label_embedding = nn.Embedding(num_embeddings=10, embedding_dim=configs.embedding_dim)
        self.w_noise = w_noise

    def forward(self, x, **kwargs):
        x = self._embeddings(x)

        label = self.label_embedding(kwargs['label'])
        x += label + self.w_noise * torch.randn_like(label).to(x.device)

        x = self._output(x)
        return x

class CondGeneratorLSH(CondGenerator):
    def __init__(self, configs):
        super(CondGeneratorLSH, self).__init__(configs)

        self.layers = nn.ModuleList([
            ReformerEncoderLayer(
                model_dim=configs.embedding_dim,
                heads=configs.heads,
                d_k=configs.d_k,
                d_v=configs.d_v,
                expansion_factor=configs.expansion_factor,
                dropout=configs.dropout,
                attn_dropout=configs.attn_dropout,
                device=configs.device,
                n_rounds=configs.n_rounds,
                max_len=configs.max_len,
                bucket_size=configs.bucket_size,
                self_attn_mask=configs.self_attn_mask,
            ) for _ in range(configs.depth)
        ])

        # Causal Mask
        self.causal_mask = torch.tril(torch.ones(2 * configs.bucket_size, 2 * configs.bucket_size)).to(configs.device)
