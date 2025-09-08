import torch
import math
import torch.nn as nn



def get_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model, requires_grad=False)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class LearnedPositionalEncoding2D(nn.Module):
    def __init__(self, height: int, width: int, d_model: int):
        super().__init__()

        self.row_embed = nn.Embedding(height, d_model)
        self.col_embed = nn.Embedding(width, d_model)
        self.height = height
        self.width = width
        self.d_model = d_model

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
           seq_len = height * width
        """
        n_tokens = x.shape[1]
        device = x.device
        rows = torch.arange(self.height, device=device)
        cols = torch.arange(self.width, device=device)

        row_pos = self.row_embed(rows)              # [H, d_model]
        col_pos = self.col_embed(cols)              # [W, d_model]

        pos = row_pos[:, None, :] + col_pos[None, :, :]

        pos = pos.view(-1, self.d_model)

        return x + pos.unsqueeze(0)[:, :n_tokens, :]  # [B, H*W, d_model]
