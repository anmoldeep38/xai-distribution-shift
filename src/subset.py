"""Fixed explanation subset and the fitting/held-out (F/H) split.

Frozen protocol (README):
- 2,000 CIFAR-10 test images, stratified 200 per class, fixed indices (seed 0).
- Split into two disjoint halves of 1,000 by clean image identity: half F (fitting) and half H (held-out).
- Every corrupted version of an image inherits its half, so an image never fits a monitor and evaluates it.
"""
import numpy as np

SUBSET_SEED = 0
N_PER_CLASS = 200
N_CLASSES = 10


def explanation_subset(test_targets, seed=SUBSET_SEED, n_per_class=N_PER_CLASS):
    """Return (indices, half) aligned and sorted by index.

    indices : int array, positions into the CIFAR-10 test set (length n_per_class * 10).
    half    : array of 'F' / 'H', the split each image belongs to (1000 / 1000).
    Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    test_targets = np.asarray(test_targets)
    idx_parts, half_parts = [], []
    for c in range(N_CLASSES):
        pool = np.where(test_targets == c)[0]
        pick = np.sort(rng.choice(pool, size=n_per_class, replace=False))
        n_f = n_per_class // 2
        h = np.array(["F"] * n_f + ["H"] * (n_per_class - n_f))
        rng.shuffle(h)  # which images of this class go to F vs H
        idx_parts.append(pick)
        half_parts.append(h)
    idx = np.concatenate(idx_parts)
    half = np.concatenate(half_parts)
    order = np.argsort(idx)
    return idx[order], half[order]


if __name__ == "__main__":
    # Sanity: the split is deterministic and balanced.
    import sys
    from pathlib import Path
    from torchvision import datasets

    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    test = datasets.CIFAR10(root, train=False, download=True)
    idx, half = explanation_subset(np.array(test.targets))
    print(f"subset size: {len(idx)}  (F={np.sum(half=='F')}, H={np.sum(half=='H')})")
    print(f"unique images: {len(set(idx.tolist()))}")
    # same seed -> identical result
    idx2, half2 = explanation_subset(np.array(test.targets))
    print("reproducible:", bool(np.array_equal(idx, idx2) and np.array_equal(half, half2)))
