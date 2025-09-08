import torch.nn as nn
from dl_proj.utils.positional_encodigs import get_positional_encoding
import torch
from my_transformers.modules import ScaledDotProduct, LSHScaledDotProduct
import os, psutil

def return_qkv(b_size, vocab_size, length, embed_dim, heads, dk, dv, device):
    emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    positional_encoding = get_positional_encoding(length, embed_dim)
    q_layer = nn.Linear(embed_dim, heads * dk)
    k_layer = nn.Linear(embed_dim, heads * dk)
    v_layer = nn.Linear(embed_dim, heads * dv)

    x = torch.randint(1, vocab_size, (b_size, length // 2))
    x = torch.cat([x, x], dim=-1)

    x = emb(x) + positional_encoding
    q = q_layer(x)
    k = k_layer(x)
    v = v_layer(x)

    q = q.reshape(b_size, length, heads, dk).to(device=device)
    k = k.reshape(b_size, length, heads, dk).to(device=device)
    v = v.reshape(b_size, length, heads, dk).to(device=device)

    return q, k, v

def return_full_heatmap(q, k, v, dk, length, device, causal_mask):
    mask = torch.tril(torch.ones(length, length)).to(device)
    prod = ScaledDotProduct(scale_factor=dk**0.5, attn_dropout=0.)
    prod(q, k, v, mask=mask if causal_mask else None)
    return prod.heatmap.detach().cpu()

def _lsh_heatmap(attn, sorted_idx, pad_dim):
    attn = attn.mean(dim=3)
    # attn shape : (rounds, batch, n_chunks, 2 * chunk_dim, 2 * chunk_dim)

    pos = 0
    max_len = sorted_idx.size(2) - pad_dim
    chunk_size = attn.size(-1) // 2
    n_chunks = attn.size(2)
    n_rounds = attn.size(0)

    padding = -torch.ones(sorted_idx.size(0), sorted_idx.size(1), chunk_size, sorted_idx.size(-1),
                          dtype=torch.int8).to(attn.device)
    sorted_idx = torch.cat((padding, sorted_idx), dim=2)

    global_attn = torch.ones(max_len, max_len).to(attn.device) * (-1e15)
    glob_att_dict = {}
    counts = torch.zeros(max_len, max_len).to(attn.device)

    for r in range(n_rounds):
        attn_single = attn[r, pos, :, :, :]  # (n_chunks, 2*chunk_size, 2*chunk_size)
        sorted_indices_single = sorted_idx[r, pos, :, pos]  - pad_dim # (max_len + pad_dim + chunk_size,)

        for i in range(n_chunks):
            start = i * chunk_size
            stop = (i + 2) * chunk_size

            chunk_indices = sorted_indices_single[start:stop]
            attn_local = attn_single[i]  # (2*chunk_size, 2*chunk_size)

            for idx, q in enumerate(chunk_indices):
                if q.item() < 0:
                    continue
                for idy, k in enumerate(chunk_indices):
                    if k.item() < 0:
                        continue

                    coordinate = (q.item(), k.item())
                    if coordinate in glob_att_dict:
                        glob_att_dict[coordinate].append(attn_local[idx, idy].item())
                    else:
                        glob_att_dict[coordinate] = [attn_local[idx, idy].item()]

                    # print(f'({q.item()}, {k.item()}) ---> {attn_local[idx, idy].item():.4f}')
                    global_attn[q, k] += attn_local[idx, idy]
                    counts[q, k] += 1

    for coordinate, values in glob_att_dict.items():
        val_tensor = torch.tensor(values)
        val_softmax = torch.softmax(val_tensor, dim=-1)
        global_attn[coordinate[0], coordinate[1]] = (val_tensor * val_softmax).sum(dim=-1)

    global_attn = global_attn.softmax(dim=-1)
    return global_attn.detach().cpu()

def return_lsh_heatmap(q, k, v, dk, dv, heads, length, n_rounds, bucket_size, device, causal_mask):
    dim = 2 * bucket_size
    mask = torch.tril(torch.ones(dim, dim)).to(device)
    prod = LSHScaledDotProduct(
        scale_factor=dk ** 0.5,
        attn_dropout=0.,
        d_k=dk,
        d_v=dv,
        heads=heads,
        n_rounds=n_rounds,
        max_len=length,
        device=device,
        bucket_size=bucket_size,
        self_attn_mask=causal_mask,
    )
    prod(q, k, v, mask=mask if causal_mask else None)
    return _lsh_heatmap(prod.heatmap, prod.sorted_idx, prod.pad_dim)

def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def get_tensor_memory(tensor):
    memory = tensor.element_size() * tensor.nelement() / (1024 ** 2)
    # print(f'Memory usage: {memory:.2f} MB')
    return memory