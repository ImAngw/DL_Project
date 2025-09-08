import torch.nn as nn
from my_transformers.sublayers import (
    MultiHeadSelfAttention,
    ReformerMultiHeadSelfAttention,
    ReversibleFeedForward
)
from revtorch.revtorch import ReversibleBlock, ReversibleSequence

# NOTE: all the classes require a tensor of the shape (B, n_words, d_model)


class FWrapper(nn.Module):
    def __init__(self, attention_module, mask=None):
        super().__init__()
        self.attn = attention_module
        self.mask = mask

    def forward(self, x):
        return self.attn(x, x, x, mask=self.mask)

class GWrapper(nn.Module):
    def __init__(self, forward_module):
        super().__init__()
        self.feed_forward = forward_module

    def forward(self, x):
        return self.feed_forward(x)


class ReversibleEncoderLayer(nn.Module):
    def __init__(self,
                 model_dim: int = 512,
                 heads: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 dropout: float = 0.,
                 ):

        super().__init__()
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v

        if model_dim % 2 != 0:
            raise ValueError("model_dim must be divisible by 2")
        self.model_dim = model_dim // 2

        self.attention = MultiHeadSelfAttention(
            model_dim=self.model_dim,
            heads=heads,
            d_k=d_k,
            d_v=d_v,
            dropout=0.,
            attn_dropout=0.,
            is_reversible=True
        )

        self.feed_forward = ReversibleFeedForward(
            model_dim=self.model_dim
        )

        f = FWrapper(self.attention, mask=None)
        g = GWrapper(self.feed_forward)
        rev_block = ReversibleBlock(f, g, split_along_dim=-1)
        rev_block = nn.ModuleList([rev_block])
        self.rev_seq = ReversibleSequence(rev_block)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        self.rev_seq.reversible_blocks[0].f_block.mask = mask
        x = self.rev_seq(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class ReversibleReformerEncoderLayer(ReversibleEncoderLayer):
    def __init__(
            self,
            n_rounds: int = 1,
            max_len: int = 256,
            bucket_size: int = 32,
            model_dim: int = 512,
            heads: int = 8,
            d_k: int = 64,
            d_v: int = 64,
            dropout: float = 0.,
            device: str = 'cpu',
            self_attn_mask: bool = False,
        ):

        super().__init__(
            model_dim=model_dim,
            heads=heads,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )

        attn_args = dict(
            model_dim=model_dim // 2,
            heads=heads,
            d_k=d_k,
            d_v=d_v,
            dropout=0.,
            attn_dropout=0.
        )

        self.attention = ReformerMultiHeadSelfAttention(
            n_rounds=n_rounds,
            max_len=max_len,
            bucket_size=bucket_size,
            device=device,
            self_attn_mask=self_attn_mask,
            **attn_args
        )

        self.rev_seq.reversible_blocks[0].f_block.attn = self.attention

