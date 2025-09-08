import torch
import torch.nn as nn
from my_transformers.local_sensitive_hashing import AngularLocalSensitiveHashing


class ScaledDotProduct(nn.Module):
    def __init__(self,
                 scale_factor: int,
                 attn_dropout: float
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(attn_dropout)
        self.heatmap = None


    def forward(self, q, k, v, mask=None):
        attention = torch.einsum("bqhd,bkhd->bhqk", [q, k]).to(q.device) / self.scale_factor

        if mask is not None:
            attention = attention.masked_fill(mask.logical_not(), float("-inf"))
            attention += torch.eye(attention.shape[-1]).to(attention.device) * (-1e5)

        attention = self.dropout(torch.softmax(attention, dim=-1))

        # Save a copy of the attention map for visualization
        heatmap = attention.mean(dim=1)
        self.heatmap = heatmap[0]
        #####################################################

        attention = torch.einsum("bhqk,bkhd->bqhd", [attention, v])  # attention shape: (B, len_q, h, d)
        return attention

class LSHScaledDotProduct(ScaledDotProduct):
    def __init__(self,
                 scale_factor: int,
                 attn_dropout: float,
                 d_k: int,
                 d_v: int,
                 heads: int,
                 n_rounds: int,
                 max_len: int,
                 bucket_size: int,
                 device: str,
                 self_attn_mask: bool
                 ):

        super().__init__(scale_factor=scale_factor, attn_dropout=attn_dropout)
        self.device = device
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.lsh = AngularLocalSensitiveHashing(d_k * heads, n_rounds, max_len, bucket_size, device)

        # Variables for performance_studies
        self.sorted_idx = None
        self.pad_dim = 0
        self.self_attn_mask = self_attn_mask
        ########################################


    def forward(self, q, k, v, mask=None):
        """
        Locality-Sensitive-Hashing Scaled Dot-Product Attention.

        Args:
            q: Query tensor of shape (batch, max_len, d_k * heads)
            k: Key tensor of shape   (batch, max_len, d_k * heads)
            v: Value tensor of shape (batch, max_len, d_v * heads)
            mask: Optional block mask to apply within each local window of size (2 * chunk_dim, 2 * chunk_dim).
                  If provided, its expected shape per-round/per-chunk broadcast is (1,1,1,1, 2*chunk_dim, 2*chunk_dim) or
                  equivalent broadcastable.

        Returns:
            Tensor of shape (batch, max_len, heads, d_v)
        """
        # INPUT SIZE
        # q, k: (batch, max_len, d_k * h)
        # v: (batch, max_len, d_v * h)

        # SORT & RESHAPE
        q, h, sorted_idx, pad_dim = self.lsh.return_sorted(q)
        k = self.lsh.sort(k, sorted_idx)
        v = self.lsh.sort(v, sorted_idx)

        q = q.view(q.shape[0], q.shape[1], q.shape[2], q.shape[3], self.heads, self.d_k)    # size: (n_round, batch, n_chunks, chunk_dim, heads, dk)
        k = k.view(k.shape[0], k.shape[1], k.shape[2], k.shape[3], self.heads, self.d_k)    # size: (n_round, batch, n_chunks, chunk_dim, heads, dk)
        v = v.view(v.shape[0], v.shape[1], v.shape[2], v.shape[3], self.heads, self.d_v)    # size: (n_round, batch, n_chunks, chunk_dim, heads, dv)


        # CHUNKING
        q, k, v, h = self.lsh.qkv_chunk(q), self.lsh.qkv_chunk(k), self.lsh.qkv_chunk(v), self.lsh.h_chunk(h)
        # q, k size: (n_round, batch, n_chunks, 2 * chunk_dim, heads, dk)
        # v size: (n_round, batch, n_chunks, 2 * chunk_dim, heads, dv)
        # h size: (n_round, batch, n_chunks, 2 * chunk_dim)


        # ATTENTION EVALUATION
        attn = torch.einsum("rbnqhd,rbnkhd-> rbnhqk", [q, k])  / self.scale_factor
        # attn size: (n_round, batch, n_chunks, heads, 2 * chunk_dim, 2 * chunk_dim


        # MASKING
        # Hash mask for each round
        h_mask = (h.unsqueeze(-1) == h.unsqueeze(3)) & (h.unsqueeze(-1) != 0)
        h_mask = h_mask.unsqueeze(3)
        h_mask = h_mask.int()


        # n: weight to avoid counting multiple times the same token
        n = h_mask * h_mask.sum(dim=4, keepdim=True)
        n = -torch.log(n.clip(min=1))

        # m: mask that takes into account all the elements with the same hash
        m = (1 - h_mask) * (-1e15)

        attn += m + n
        # attn size: (n_round, batch, n_chunks, heads, 2 * chunk_dim, 2 * chunk_dim)


        if mask is not None:
            # mask size must be: (4 * bucket_size, 4 * bucket_size) = (2 * chunk_dim, 2 * chunk_dim)
            attn += (1 - mask) * (-1e15)

        if self.self_attn_mask:
            # Principal diagonal mask (to avoid self attention)
            i_mask = torch.eye(h_mask.size(-1), device=self.device)
            i_mask = i_mask.view(1, 1, 1, 1, h_mask.size(-1), h_mask.size(-1))
            attn += i_mask * (-1e5)

        # Save a copy of the attention map for visualization
        self.heatmap = attn
        self.sorted_idx = sorted_idx
        self.pad_dim = pad_dim


        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn = torch.einsum('rbnhml, rbnlhd -> rbnmhd', [attn, v])
        # attn size: (n_round, batch, n_chunks, 2 * chunk_dim, heads, dv)

        # Remove the extra dimensions over the chunk dimension (that turns out from padding in the chunking process)
        attn = attn[:, :, :, int(attn.shape[3]/2):, :, :]

        # attn size: (n_round, batch, n_chunks, chunk_dim, heads, dv)
        attn = attn.reshape(attn.shape[0], attn.shape[1], attn.shape[2] * attn.shape[3], self.heads, self.d_v)

        # attn size: (n_round, batch, max_len, heads, dv)
        attn = self.lsh.original_sort(attn, sorted_idx)
        if pad_dim != 0:
            attn = attn[:, :, pad_dim:, :, :]

        # OUTPUT
        # attn size: (batch, max_len, heads, dv)
        return attn.sum(dim=0)

