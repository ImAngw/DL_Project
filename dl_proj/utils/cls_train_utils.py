from my_custom_ai.utils.train_utils import FunctionContainer
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from dl_proj.models.classification_model import  ClassificationModel, ClassificationModelLSH
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Subset


class ClassificationContainer(FunctionContainer):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.criterion = nn.CrossEntropyLoss()

    def batch_extractor(self, batch, *args, **kwargs):
        b, y = batch
        y = y.squeeze(1)
        return b.to(self.configs.device), y.to(self.configs.device)

    def loss_function(self, model_output, y, *args, **kwargs):
        loss = self.criterion(model_output, y)
        return loss

    def validation_performance(self, model, loader, *args, **kwargs):
        total = 0
        corrects = 0
        scores = {}

        for idx, batch in enumerate(loader):
            batch, y = self.batch_extractor(batch)
            output = model(batch)
            predictions = torch.argmax(output, dim=-1)
            correct = torch.sum(predictions == y)

            total += y.size(0)
            corrects += correct.item()

        score = corrects / total
        scores['score'] = score
        return scores

    def test_performance(self, model, loader, pbar, *args, **kwargs):
        pass

class TransformedImageDataset(Dataset):
    """
    Dataset wrapper that converts an image dataset into token-like sequences for two tasks:
    classification (cls) and next-token generation (gen).

    This class assumes input images are already quantized to integer levels in [0, vocab_size-1].
    Quantization is typically applied via the get_loaders transform pipeline.

    Parameters:
    - original_dataset: a torch.utils.data.Dataset yielding (image, label) pairs where
      image is a tensor of shape (C, H, W) with integer values (quantized) and label is an int.
    - vocab_size (int): number of discrete pixel/patch values. Two special tokens are reserved for
      generation mode: SOS=vocab_size and EOS=vocab_size+1.
    - patch_size (int, default=1): spatial patch size used to reshape images. For classification,
      the image is partitioned into non-overlapping patches of size (patch_size x patch_size).
    - patch_on_channels (bool, default=False):
      - If True (classification only): each patch flattens across channels and spatial dims, yielding a
        token per spatial patch of length C*patch_size*patch_size. Output shape per image:
        (n_patches, C*patch_size*patch_size).
      - If False: treat each channel independently, yielding (n_patches*C, patch_size*patch_size).
    - task (str, default='cls'): either 'cls' for classification tokenization or 'gen' for next-token generation.

    Outputs from __getitem__(idx):
    - task == 'cls': returns (img_tokens, label)
        - img_tokens (FloatTensor):
          - If patch_on_channels is True: shape (n_patches, C*patch_size*patch_size), values in [0,1].
          - Else: shape (n_patches*C, patch_size*patch_size), values in [0,1].
        - label (LongTensor): shape (1,), integer class label.
    - task == 'gen': returns (input_seq, target_seq, label)
        - input_seq (LongTensor): flattened image sequence with SOS inserted at the start; shape (L,).
        - target_seq (LongTensor): input_seq shifted by one with EOS at the end; shape (L,).
        - label (LongTensor): shape (1,), integer class label.

    Notes:
    - For classification, tokens are normalized to float in [0,1] by dividing by 255.0 after reshaping, while
      keeping their discrete structure. For generation, tokens remain integers (long dtype), as they represent
      categorical vocabulary indices including special tokens.
    - The number of patches n_patches = (H//patch_size) * (W//patch_size).
    """
    def __init__(self, original_dataset, vocab_size, patch_size=1, patch_on_channels=False, task='cls'):
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.patch_on_channels = patch_on_channels
        self.task = task

        self.imgs = []
        self.targets = []
        self.labels = []

        if task == 'gen':
            self._generation_transform(original_dataset)
        elif task == 'cls':
            self._classification_transform(original_dataset)
        else:
            raise NotImplementedError


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.task == 'gen':
            return self.imgs[idx], self.targets[idx], self.labels[idx]
        elif self.task == 'cls':
            return self.imgs[idx], self.labels[idx]
        else:
            raise NotImplementedError

    def _classification_transform(self, original_dataset):
        for b, y in original_dataset:
            channels, height, width = b.shape
            n_patches = (height // self.patch_size) * (width // self.patch_size)

            img = b.reshape(channels, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size).to(dtype=torch.long)
            img = img.permute(1, 3, 0, 2, 4)
            if self.patch_on_channels:
                img = img.reshape(n_patches, channels * self.patch_size * self.patch_size)
                # shape: (n_patches, c * patch_dim * patch_dim)
            else:
                img = img.reshape(n_patches * channels, self.patch_size * self.patch_size)
                # shape: (n_patches * c, patch_dim * patch_dim)
            img = img.float() / 255.0

            self.imgs.append(img)
            self.labels.append(torch.tensor([y]))

    def _generation_transform(self, original_dataset):
        sos = torch.tensor([self.vocab_size])
        eos = torch.tensor([self.vocab_size + 1])

        for b, y in original_dataset:
            channels, height, width = b.shape
            b = b.reshape(channels * height * width).to(dtype=torch.long)

            ext_img = torch.cat([sos, b[:-1], eos], dim=0)
            img = ext_img[:-1]
            target = ext_img[1:]

            self.imgs.append(img)
            self.targets.append(target)
            self.labels.append(torch.tensor([y]))

def get_cls_model(configs, n_patches, patch_dim):
    if configs.attn_type == 'full':
        model = ClassificationModel(configs=configs, n_patch=n_patches, patch_dim=patch_dim).to(configs.device)
    elif configs.attn_type == 'lsh':
        model = ClassificationModelLSH(configs=configs, n_patch=n_patches, patch_dim=patch_dim).to(configs.device)
    else:
        raise ValueError('Unknown attention type')
    return model

def get_loaders(configs):
    """
    Build training and test DataLoaders for image classification or generation tasks,
    converting images into quantized token-like representations compatible with
    TransformedImageDataset.

    This utility performs the following steps:
    - Chooses dataset by configs.dataset ('mnist' -> grayscale, else CIFAR10 -> RGB).
    - Computes a square resize dimension so that channels * H * W ~= configs.max_len.
    - Applies a transform pipeline: ToTensor -> Resize(dim, dim) -> pixel quantization
      into configs.vocab_size discrete levels (0..vocab_size-1).
    - Filters samples to only those whose label is in configs.labels.
    - Wraps the subset with TransformedImageDataset according to configs.task ('cls' or 'gen'),
      configs.patch_size, and configs.patch_on_channels.
    - Creates train and test DataLoaders using configs.batch_size.

    Parameters (from configs expected/used here):
    - dataset (str): 'mnist' or 'cifar10' (any non-'mnist' value selects CIFAR10 here).
    - labels (Iterable[int]): list of class labels to include.
    - vocab_size (int): number of quantization levels for pixels; also acts as SOS index in 'gen'
      mode with EOS=vocab_size+1 inside TransformedImageDataset.
    - max_len (int): target flattened image length (channels * H * W) used to compute resize dims.
    - patch_size (int): spatial patch size used by TransformedImageDataset when task == 'cls'.
    - patch_on_channels (bool): patching strategy for classification tokenization.
    - task (str): 'cls' for classification or 'gen' for next-token generation.
    - batch_size (int): batch size for the DataLoaders.

    Returns:
    - train_loader (DataLoader): over TransformedImageDataset subset of the chosen dataset.
    - test_loader (DataLoader): same for the test split.
    - n_patches (int or None): number of tokens per sample for classification; None for 'gen'.
    - patch_dim (int or None): token dimensionality for classification; None for 'gen'.

    Notes:
    - MNIST is treated as single-channel (C=1); CIFAR10 as three-channel (C=3).
    - Quantization maps input pixel values in [0, 255] to integer bins [0, vocab_size-1] via
      floor(pixel / (256/levels)).
    - For configs.task == 'cls', (n_patches, patch_dim) are inferred from the first sample of the
      transformed training subset as train_subset[0][0].shape.
    """
    class QuantizePixels:
        def __init__(self, levels=32):
            self.levels = levels

        def __call__(self, img_tensor):
            img_255 = img_tensor * 255.0
            quantized = torch.floor(img_255 / (256 / self.levels)).int()
            return quantized

    channels = 1 if configs.dataset == 'mnist' else 3
    dim = int((configs.max_len / channels) ** 0.5)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((dim, dim)),
        QuantizePixels(levels=configs.vocab_size)
    ])

    if configs.dataset == 'mnist':
        train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_indices = [i for i, label in enumerate(train.targets) if label in configs.labels]
    train_subset = Subset(train, train_indices)
    train_subset = TransformedImageDataset(
        train_subset,
        configs.vocab_size,
        configs.patch_size,
        configs.patch_on_channels,
        configs.task
    )

    test_indices = [i for i, label in enumerate(test.targets) if label in configs.labels]
    test_subset = Subset(test, test_indices)
    test_subset = TransformedImageDataset(
        test_subset,
        configs.vocab_size,
        configs.patch_size,
        configs.patch_on_channels,
        configs.task
    )

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=configs.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=configs.batch_size, shuffle=False)

    if configs.task == 'cls':
        n_patches, patch_dim = train_subset[0][0].shape
    else:
        n_patches, patch_dim = None, None

    return train_loader, test_loader, n_patches, patch_dim