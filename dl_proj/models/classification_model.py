import torch
import torch.nn as nn
from my_transformers.layers import ReformerEncoderLayer, EncoderLayer


class ClassificationModel(nn.Module):
    def __init__(self,
                 configs,
                 n_patch,       # n patches for each image
                 patch_dim      # n pixels for each patch
        ):

        super().__init__()
        self.config = configs
        # vector for positional embedding
        self.register_buffer('pos', torch.arange(0, n_patch, dtype=torch.long).unsqueeze(0))
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, configs.embedding_dim))

        # MODULES
        self.patch_embedding = nn.Linear(patch_dim, configs.embedding_dim)
        self.pos_embedding = nn.Embedding(n_patch, configs.embedding_dim)
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
            nn.GELU(),
            nn.Linear(configs.embedding_dim, len(configs.labels))
        )

    def forward(self, x):

        x = self.patch_embedding(x)     # input embedding
        pos_emb = self.pos_embedding(self.pos)  # positional embedding
        x = x + pos_emb     # sum all together

        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)     # add the cls token

        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.out(x[:, 0, :])    # return logits of CLS token
        return x

class ClassificationModelLSH(ClassificationModel):
    def __init__(self, configs, n_patch, patch_dim):
        super().__init__(configs, n_patch, patch_dim)

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
