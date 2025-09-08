from torch.utils.data import Dataset
import torch



class SyntheticDataset(Dataset):
    """
    Synthetic sequence-to-sequence dataset for conditional generation/denoising.

    Each sample is constructed by concatenating two sequences of length `seq_len`:
    - Original half: [SOS] + (seq_len-1) random tokens from [1, vocab_size-1].
    - Corrupted half: [SOS] + token-wise mixture of the original tokens with probability `p` replaced by
      tokens drawn uniformly from [1, vocab_size-1].

    The final input sequence x therefore has length 2*seq_len and is:
        x = concat(original_with_SOS, corrupted_with_SOS)

    The target sequence y has the same length as x and is designed for next-token training only on the
    second half (the first half is ignored):
        - First seq_len positions are filled with -1 (ignore_index) so the loss is not computed there.
        - Next (seq_len-1) positions contain the original tokens (without SOS) that correspond to the
          intended outputs given the corrupted half as input.
        - The last position is EOS=vocab_size.

    This setup encourages a model to reconstruct the original clean sequence from its corrupted version,
    conditioned on the SOS token at the start of the second half.

    Parameters
    - seq_len (int): length of each half including the leading SOS; must be >= 2.
    - n_samples (int): number of unique samples to generate.
    - vocab_size (int, default=10): number of discrete tokens. Reserved tokens:
        SOS=0, EOS=vocab_size. Regular tokens are in [1, vocab_size-1].
    - p (float, default=0.0): per-position corruption probability used to mix the corrupted half.

    Returns from __getitem__(idx)
    - x (LongTensor): shape (2*seq_len,), the concatenated input sequence.
    - y (LongTensor): shape (2*seq_len,), training targets with the first half set to -1 (ignore_index)
      and the second half equal to the original tokens followed by EOS.

    Notes
    - Uniqueness: samples are regenerated until `n_samples` unique sequences are collected.
    - Dtypes: tensors are torch.long. Use loss functions that support ignore_index=-1 (e.g., CrossEntropyLoss).
    """
    def __init__(self, seq_len, n_samples, vocab_size=10, p=0.):
        super().__init__()
        self.p = p
        self.seq_len = seq_len
        self.sos = 0
        self.eos = vocab_size
        self.n_samples = n_samples
        self.vocab_size = vocab_size
        self.dataset = self._build_dataset()


    def _build_dataset(self):
        dataset = []
        list_of_seq = []

        while len(list_of_seq) < self.n_samples:
            x = torch.randint(1, self.vocab_size, (self.seq_len - 1,))

            mixed = torch.randint(1, self.vocab_size, (self.seq_len - 1,))
            mask = (torch.rand_like(x.float()) < self.p)
            x_mixed = torch.where(mask, mixed, x)

            x = torch.cat((torch.tensor([self.sos]), x))
            x_mixed = torch.cat((torch.tensor([self.sos]), x_mixed))

            x = torch.cat((x, x_mixed))
            if x.tolist() not in list_of_seq:
                dataset.append(x)
                list_of_seq.append(x.tolist())
        return dataset

    def __getitem__(self, idx):
        x = self.dataset[idx]
        y = self.dataset[idx][1:self.seq_len]
        y = torch.cat([torch.tensor([-1] * self.seq_len), y, torch.tensor([self.eos])])
        return x, y

    def __len__(self):
        return len(self.dataset)
