"""CIFAR-10 / CIFAR-10-C loading. All tensors are in [0, 1]; normalisation happens inside the model."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CORRUPTIONS = [  # the 15 benchmark corruptions of Hendrycks & Dietterich (2019)
    "gaussian_noise", "shot_noise", "impulse_noise",          # noise
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",  # blur
    "snow", "frost", "fog", "brightness",                     # weather
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",  # digital
]
FAMILIES = {
    "noise": CORRUPTIONS[0:3],
    "blur": CORRUPTIONS[3:7],
    "weather": CORRUPTIONS[7:11],
    "digital": CORRUPTIONS[11:15],
}
N_PER_SEVERITY = 10_000


def train_val_split(n=50_000, n_val=5_000, seed=0):
    """Fixed split of the CIFAR-10 training set. Same seed -> same indices."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def cifar10_loaders(root="data", batch_size=128, split_seed=0, num_workers=2):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()

    train_full = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    train_eval = datasets.CIFAR10(root, train=True, download=False, transform=test_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    tr_idx, va_idx = train_val_split(len(train_full), seed=split_seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    loaders = {
        "train": DataLoader(Subset(train_full, tr_idx), shuffle=True, drop_last=True,
                            persistent_workers=num_workers > 0, **kw),
        "val": DataLoader(Subset(train_eval, va_idx), shuffle=False, **kw),
        "test": DataLoader(test, shuffle=False, **kw),
    }
    return loaders, {"train": tr_idx, "val": va_idx}


class CIFAR10C(Dataset):
    """One corruption, one or more severities (1-5). Loaded lazily via memory-map."""

    def __init__(self, root, corruption, severities=(1, 2, 3, 4, 5)):
        root = Path(root)
        self.images = np.load(root / f"{corruption}.npy", mmap_mode="r")  # (50000, 32, 32, 3) uint8
        self.labels = np.load(root / "labels.npy")
        self.index = np.concatenate([
            np.arange((s - 1) * N_PER_SEVERITY, s * N_PER_SEVERITY) for s in severities
        ])
        self.corruption = corruption

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = self.index[i]
        img = torch.from_numpy(np.ascontiguousarray(self.images[j])).permute(2, 0, 1).float() / 255.0
        severity = j // N_PER_SEVERITY + 1
        clean_idx = j % N_PER_SEVERITY  # position in the CIFAR-10 test set
        return img, int(self.labels[j]), severity, clean_idx
