import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularLocalSensitiveHashing(nn.Module):
    def __init__(
            self,
            hidden_dim:int,
            n_rounds:int,
            max_len:int,
            bucket_size:int,
            device:str,
            max_n_rounds:int=16
    ):

        super().__init__()
        if max_len < bucket_size:
            raise ValueError("max_len must be greater than bucket_size")
        if max_len % bucket_size != 0:
            raise ValueError("max_len must be divisible by bucket_size")

        self.device = device
        self.chunk_dim = bucket_size
        self.n_rounds = n_rounds
        self.max_n_rounds = max_n_rounds
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.n_hashes = max_len // bucket_size
        random_matrix = torch.stack([
            torch.linalg.qr(torch.randn(hidden_dim, hidden_dim, requires_grad=False))[0][:self.n_hashes]
            for _ in range(self.max_n_rounds)
        ]).to(device)

        self.register_buffer("random_matrix", random_matrix)
        self.positions = torch.arange(max_len).to(device)



    def return_sorted(self, x):
        """
        Given a tensor x, the function computes the hashes and sorts the elements of x based on those hash values and sequence positions.

        :param x: a tensor of size (batch_size, max_len, hidden_dim)
        :return:
            x_sorted: sorted tensor (divided in chunks) of size (n_rounds, batch_size, n_chunks, chunk_dim, hidden_dim)
            sorted_hashes: A tensor that stores the hash values corresponding to the elements of x_sorted
            sorted_indices: A tensor that holds the original indices, reordered according to the hashing operations.
            pad_dim: padding dimension
        """

        x = x.expand(self.n_rounds, *x.shape)   # x: (rounds, batch, max_len, hidden_dim)
        x_norm = F.normalize(x, dim=-1)

        # Make a random permutation in order to use different self.random_matrix columns
        perm = torch.randperm(self.random_matrix.shape[0])
        rand_matrix_perm = self.random_matrix[perm, :, :]

        # Hash
        rotation = torch.einsum('rbld,rcd->rblc', x_norm, rand_matrix_perm[:self.n_rounds, :, :])
        hashes = torch.argmax(torch.cat((rotation, -rotation), dim=3), dim=3) + 1

        # Sorting
        sort_keys = hashes * hashes.size(2) + self.positions[:x.size(2)]       # The sum makes each key unique, and sort_keys
                                                                               # can also be ordered according to its position.
        sorted_indices = torch.argsort(sort_keys, dim=2, stable=True)

        # if length % chunk_dim != 0 add zero padding
        if hashes.size(2) % self.chunk_dim != 0:
            pad_dim = self.chunk_dim - (hashes.size(2) % self.chunk_dim)
            hash_pad = torch.zeros(hashes.size(0), hashes.size(1), pad_dim, dtype=torch.int8, device=self.device)
            idx_pad = torch.arange(0, pad_dim, device=self.device)
            idx_pad = idx_pad.expand(sorted_indices.size(0), sorted_indices.size(1), pad_dim)
            x_pad = torch.zeros(x.size(0), x.size(1), pad_dim, x.size(3), device=self.device)

            hashes = torch.cat((hash_pad, hashes), dim=2)
            sorted_indices += pad_dim
            sorted_indices = torch.cat((idx_pad, sorted_indices), dim=2)
            x = torch.cat((x_pad, x), dim=2)
        else:
            pad_dim = 0


        sorted_indices = sorted_indices.unsqueeze(-1)
        x_sorted = torch.take_along_dim(x, sorted_indices, dim=2)

        # Divide in chunks
        x_sorted = x_sorted.view(x_sorted.size(0), x_sorted.size(1), (x_sorted.size(2) // self.chunk_dim), self.chunk_dim, x_sorted.size(3))
        sorted_hashes = hashes.sort(dim=2).values
        sorted_hashes = sorted_hashes.view(sorted_hashes.size(0), sorted_hashes.size(1), (sorted_hashes.size(2) // self.chunk_dim), self.chunk_dim)


        # OUTPUT SIZES
        #      x_sorted: (n_round, batch, n_chunks, chunk_dim, hidden_dim)
        # sorted_hashes: (n_round, batch, n_chunks, chunk_dim)
        # sorted_indices: (n_round, batch, max_len, 1)
        return x_sorted, sorted_hashes, sorted_indices, pad_dim


    def sort(self, x, sorted_idx):
        """
        Given a tensor x and a tensor of indices sorted_idx, reorder x based on the provided indices and divide x in chunks.

        :param x: tensor of size (batch_size, max_len, hidden_dim)
        :param sorted_idx: tensor of size (n_rounds, batch_size, max_len, 1)
        :return:
            x_sort: sorted tensor of size (n_rounds, batch_size, n_chunks, chunk_dim, hidden_dim)
        """
        # INPUT SIZE
        # x: (batch, max_len, hidden_dim)
        # sorted_indices: (n_round, batch, max_len, 1)

        # if length % chunk_dim != 0 add zero padding
        if x.size(1) % self.chunk_dim != 0:
            pad_dim = self.chunk_dim - (x.size(1) % self.chunk_dim)
            pad_tensor = torch.zeros(x.size(0), pad_dim, x.size(2), device=self.device)
            x = torch.cat((pad_tensor, x), dim=1)

        x = x.expand(self.n_rounds, *x.shape)

        # Divide in chunks
        x_sort = torch.take_along_dim(x, sorted_idx, dim=2)
        x_sort = x_sort.view(x_sort.size(0), x_sort.size(1), (x_sort.size(2) // self.chunk_dim), self.chunk_dim, x_sort.size(3))

        # OUTPUT SIZE
        # x_sort: (n_round, batch, n_chunks, chunk_dim, hidden_dim)
        return x_sort

    @staticmethod
    def original_sort(x, sorted_idx):
        """
        Restore the original order of tensor x using the index tensor sorted_idx.

        :param x: tensor of size (n_rounds, batch_size, max_len, heads, dv)
        :param sorted_idx: tensor of size (n_rounds, batch_size, max_len, 1)
        :return: x_sort: tensor of size (n_rounds, batch_size, max_len, heads, dv)
        """

        # INPUT SIZE
        # x size: (n_round, batch, max_len, heads, dv)
        # sorted_idx size: (n_round, batch, max_len, 1)

        rev_idx = torch.argsort(sorted_idx, dim=2, stable=True).unsqueeze(-1)
        x_sort = torch.take_along_dim(x, rev_idx, dim=2)

        # OUTPUT
        # x_sorted size: (n_round, batch, max_len, heads, dv)
        return x_sort


    @staticmethod
    def old_qkv_chunk(x_sorted):
        # INPUT SIZES
        # x_sorted: (n_round, batch, n_chunks, chunk_dim, heads, hidden_dim)

        x_prev = torch.zeros_like(x_sorted[:, :, :, :1])
        x_shifted = torch.cat([x_prev, x_sorted[:, :, :, :-1]], dim=3)
        x_chunk = torch.cat([x_shifted, x_sorted], dim=3)


        # OUTPUT SIZES
        # x_chunk: (n_round, batch, n_chunks, 2 * chunk_dim, hidden_dim)
        return x_chunk

    @staticmethod
    def qkv_chunk(x_sorted):
        """
        Builds the chunks on which attention will be computed by concatenating each chunk to the previous one
        (the first chunk is concatenated with a chunk of all zeros).

        :param x_sorted: tensor of size (n_rounds, batch_size, n_chunks, chunk_dim, heads, hidden_dim)
        :return: x_chunk: tensor of size (n_rounds, batch_size, n_chunks, 2 * chunk_dim, hidden_dim)
        """
        # INPUT SIZES
        # x_sorted: (n_round, batch, n_chunks, chunk_dim, heads, hidden_dim)
        r, b, n, c, h, d = x_sorted.shape

        # Padding for the first chunk
        first_padding = torch.zeros(r, b, 1, c, h, d, device=x_sorted.device, dtype=x_sorted.dtype)
        x_with_padding = torch.cat([first_padding, x_sorted], dim=2)  # (r, b, n+1, c, h, d)

        left = x_with_padding[:, :, :-1]  # (r, b, n, c, h, d)
        right = x_with_padding[:, :, 1:]  # (r, b, n, c, h, d)

        x_chunk = torch.cat([left, right], dim=3)  # (r, b, n, 2c, h, d)

        # OUTPUT SIZES
        # x_chunk: (n_round, batch, n_chunks, 2 * chunk_dim, hidden_dim)
        return x_chunk

    @staticmethod
    def h_chunk(h_sorted):
        """
        Builds the chunks (for the hash tensor) by concatenating each chunk to the previous one
        (the first chunk is concatenated with a chunk of all zeros).
        :param h_sorted: hash tensor of size (n_rounds, batch_size, n_chunks, chunk_dim)
        :return:  h_chunk: tensor of size (n_rounds, batch_size, n_chunks, 2 * chunk_dim)
        """
        # INPUT SIZES
        # h_sorted: (n_round, batch, n_chunks, chunk_dim)

        h_prev = F.pad(h_sorted[:, :, :-1], (0, 0, 1, 0, 0, 0))
        h_chunk = torch.cat([h_prev, h_sorted], dim=3)

        # OUTPUT SIZES
        # h_chunk: (n_round, batch, n_chunks, 2 * chunk_dim)
        return h_chunk


