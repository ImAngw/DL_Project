import torch.nn as nn
from my_transformers.modules import ScaledDotProduct, LSHScaledDotProduct
import torch.nn.functional as F





class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 model_dim: int,
                 heads: int,
                 d_k: int,
                 d_v: int,
                 dropout: float,
                 attn_dropout: float,
                 is_reversible: bool = False,
                 ):

        super().__init__()
        self.d = model_dim
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.attn_dropout = attn_dropout
        self.is_reversible = is_reversible
        self.values = nn.Linear(model_dim, d_v * heads, bias=False)
        self.keys = nn.Linear(model_dim, d_k * heads, bias=False)
        self.queries = nn.Linear(model_dim, d_k * heads, bias=False)
        self.fc_out = nn.Linear(d_v * heads, model_dim)

        self.attention = ScaledDotProduct(scale_factor=d_k ** 0.5, attn_dropout=attn_dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, queries, keys, values, mask=None):
        b_size = values.shape[0]
        query_len = queries.shape[1]

        keys = self.keys(keys)
        queries = self.queries(queries)
        values = self.values(values)


        # Projection for Linformer, else it is the identity function
        keys, values = self.project_kv(keys, values)

        # Reshape module
        queries, keys, values = self.reshape_qkv(queries, keys, values)

        attention = self.attention(queries, keys, values, mask)
        attention = attention.reshape(b_size, query_len, self.d_v * self.heads)    # attention shape: (B, len_q, h*d)


        attention = self.fc_out(attention)
        if not self.is_reversible:
            attention = self.dropout(self.fc_out(attention))

        return attention                # attn shape: (B, len_q, d)


    def project_kv(self, keys, values):
        # Identity function for standard attention
        return keys, values

    def reshape_qkv(self, q, k, v):
        values = v.view(v.shape[0], v.shape[1], self.heads, self.d_v)
        keys = k.view(k.shape[0], k.shape[1], self.heads, self.d_k)
        queries = q.view(q.shape[0], q.shape[1], self.heads, self.d_k)
        return queries, keys, values

class ReformerMultiHeadSelfAttention(MultiHeadSelfAttention):
    def __init__(self, n_rounds:int, max_len:int, bucket_size:int, device: str, self_attn_mask: bool, **kwargs):
        super().__init__(**kwargs)
        self.keys = self.queries
        self.n_rounds = n_rounds
        self.max_len = max_len
        self.bucket_size = bucket_size
        self.device = device

        self.attention = LSHScaledDotProduct(
            scale_factor=self.d_k ** 0.5,
            attn_dropout=self.attn_dropout,
            d_k=self.d_k,
            d_v=self.d_v,
            heads=self.heads,
            n_rounds=self.n_rounds,
            max_len=self.max_len,
            bucket_size=self.bucket_size,
            device=self.device,
            self_attn_mask=self_attn_mask
        )


    def reshape_qkv(self, q, k, v):
        return q, k, v

class FeedForward(nn.Module):
    def __init__(self,
                 model_dim: int,
                 expansion_factor: int,
                 dropout: float,
                 ):

        super().__init__()
        self.linear1 = nn.Linear(model_dim, model_dim * expansion_factor)
        self.linear2 = nn.Linear(model_dim * expansion_factor, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(self.linear2(x))
        return x

class ReversibleFeedForward(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, model_dim, bias=False)
        self.linear2 = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x
