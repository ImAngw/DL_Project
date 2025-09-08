import torch.nn as nn
from my_transformers.sublayers import (
    MultiHeadSelfAttention,
    ReformerMultiHeadSelfAttention,
    FeedForward
)

# NOTE: all the classes require a tensor of the shape (B, n_words, d_model)


# STANDARD TRANSFORMER LAYERS
class EncoderLayer(nn.Module):
    def __init__(
            self,
            model_dim: int = 512,
            heads: int = 8,
            d_k: int = 64,
            d_v: int = 64,
            expansion_factor: int = 4,
            dropout: float = 0.1,
            attn_dropout: float = 0.1,
        ):

        super().__init__()
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.model_dim = model_dim
        self.attn_norm_layer = nn.LayerNorm(model_dim)
        self.ff_norm_layer = nn.LayerNorm(model_dim)



        self.attention = MultiHeadSelfAttention(
            model_dim=self.model_dim,
            heads=heads,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            attn_dropout=attn_dropout
        )

        self.feed_forward = FeedForward(
            model_dim=self.model_dim,
            dropout=dropout,
            expansion_factor=expansion_factor,
        )

    def forward(self, x, mask=None):
        x = self.attn_norm_layer(self.attention(x, x, x, mask) + x)
        x = self.ff_norm_layer(self.feed_forward(x) + x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,
                 model_dim: int = 512,
                 heads: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 expansion_factor: int = 4,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.1,
                 ):

        super().__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.attn_dropout = attn_dropout


        self.self_attn = MultiHeadSelfAttention(
            model_dim=model_dim,
            heads=heads,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            attn_dropout=attn_dropout
        )

        self.enc_attn = MultiHeadSelfAttention(
            model_dim=model_dim,
            heads=heads,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            attn_dropout=attn_dropout
        )

        self.feed_forward = FeedForward(model_dim=model_dim, dropout=dropout, expansion_factor=expansion_factor)

    def forward(self, decoder_input, encoder_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        x = self.self_attn(decoder_input, decoder_input, decoder_input, slf_attn_mask)
        x = self.enc_attn(x, encoder_output, encoder_output, dec_enc_attn_mask)
        x = self.feed_forward(x)
        return x



# REFORMERS LAYERS
class ReformerEncoderLayer(EncoderLayer):
    def __init__(self,
                 n_rounds: int=1,
                 max_len: int=256,
                 bucket_size: int=32,
                 model_dim: int = 512,
                 heads: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 expansion_factor: int = 4,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.1,
                 device: str = 'cpu',
                 self_attn_mask: bool = False,
                ):

        super().__init__(
            model_dim=model_dim,
            heads=heads, d_k=d_k, d_v=d_v, expansion_factor=expansion_factor,
            dropout=dropout, attn_dropout=attn_dropout
        )

        attn_args = dict(
            model_dim=model_dim,
            heads=heads,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            attn_dropout=attn_dropout
        )


        self.attention = ReformerMultiHeadSelfAttention(
            n_rounds=n_rounds,
            max_len=max_len,
            bucket_size=bucket_size,
            device=device,
            self_attn_mask=self_attn_mask,
            **attn_args
        )

class ReformerDecoderLayer(DecoderLayer):
    def __init__(self,
                 n_rounds: int=1,
                 max_len: int=256,
                 bucket_size: int=32,
                 model_dim: int = 512,
                 heads: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 expansion_factor: int = 4,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.1,
                 device: str = 'cpu',
                 self_attn_mask: bool = False,
                 ):

        super().__init__(model_dim=model_dim, heads=heads, d_k=d_k, d_v=d_v, expansion_factor=expansion_factor,
                         dropout=dropout, attn_dropout=attn_dropout)

        attn_args = dict(model_dim=model_dim,
                         heads=heads,
                         d_k=d_k,
                         d_v=d_v,
                         dropout=dropout,
                         attn_dropout=attn_dropout)

        self.self_attn = ReformerMultiHeadSelfAttention(
            n_rounds=n_rounds,
            max_len=max_len,
            bucket_size=bucket_size,
            device=device,
            self_attn_mask=self_attn_mask,
            **attn_args
        )

        self.enc_attn = ReformerMultiHeadSelfAttention(
            n_rounds=n_rounds,
            max_len=max_len,
            bucket_size=bucket_size,
            device=device,
            self_attn_mask=self_attn_mask,
            **attn_args
        )
