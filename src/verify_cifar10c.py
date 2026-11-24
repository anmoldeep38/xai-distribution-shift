"""Check a CIFAR-10-C download and (optionally) evaluate a checkpoint on one corruption.

Usage:
    python src/verify_cifar10c.py --c10c-dir data/CIFAR-10-C --corruption gaussian_noise \
        --ckpt checkpoints/resnet18_seed0.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CORRUPTIONS, N_PER_SEVERITY, CIFAR10C  # noqa: E402
from models import ResNet18CIFAR  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c10c-dir", default="data/CIFAR-10-C")
    p.add_argument("--corruption", default="gaussian_noise")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data", default="data")
    args = p.parse_args()
    root = Path(args.c10c_dir)

    # 1. Files present?
    missing = [c for c in CORRUPTIONS + ["labels"] if not (root / f"{c}.npy").exists()]
    print(f"benchmark files missing: {missing or 'none'}")

    # 2. Shapes, dtype and label alignment with the clean CIFAR-10 test set
    imgs = np.load(root / f"{args.corruption}.npy", mmap_mode="r")
    labels = np.load(root / "labels.npy")
    print(f"{args.corruption}: shape={imgs.shape} dtype={imgs.dtype}; labels shape={labels.shape}")
    assert imgs.shape == (5 * N_PER_SEVERITY, 32, 32, 3) and imgs.dtype == np.uint8
    clean = datasets.CIFAR10(args.data, train=False, download=True)
    clean_y = np.array(clean.targets)
    aligned = all((labels[s * N_PER_SEVERITY:(s + 1) * N_PER_SEVERITY] == clean_y).all() for s in range(5))
    print(f"labels aligned with CIFAR-10 test order at every severity: {aligned}")
    # Pixel distance to the clean image should grow with severity if the ordering is right.
    clean_x = clean.data[:1000].astype(np.float32)
    for s in range(5):
        d = np.abs(imgs[s * N_PER_SEVERITY:s * N_PER_SEVERITY + 1000].astype(np.float32) - clean_x).mean()
        print(f"  severity {s + 1}: mean |corrupted - clean| = {d:.2f}")

    if not args.ckpt:
        return

    # 3. Accuracy of the checkpoint per severity
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet18CIFAR().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()
    for s in range(1, 6):
        ds = CIFAR10C(root, args.corruption, severities=(s,))
        correct = 0
        with torch.no_grad():
            for x, y, _, _ in DataLoader(ds, batch_size=500):
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        print(f"  {args.corruption} severity {s}: acc = {correct / len(ds):.4f}")


if __name__ == "__main__":
    main()
