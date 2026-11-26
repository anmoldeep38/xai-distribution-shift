"""Local heatmap drift and the output-consistency baseline (frozen protocol, Track A).

Operational drift (needs no clean image):
    D(x) = (1/K) * sum_k  d( A(x), A(T_k(x)) )
    T_k  = x + Gaussian noise (sigma = 0.01), clipped to [0,1]; K = 8; seed fixed per image.
    A    = IG or Grad-CAM map, target = the class predicted for x (same target for all K copies).
    d    = 1 - Spearman (primary), normalised L1, cosine distance (secondary).

Output-consistency baseline (same K perturbations, after Kalita & Singh 2026):
    LS(x) = fraction of the K copies whose predicted class equals the class predicted for x.
    CD(x) = mean_k | p_hat(x) - p_hat(T_k(x)) |,  p_hat = softmax prob of x's predicted class.

All drift values are per image. Higher drift = less stable explanation.
"""
import numpy as np
import torch

from attributions import compute_maps

SIGMA = 0.01   # frozen: 4x milder than CIFAR-10-C's mildest Gaussian noise
K = 8          # frozen: number of perturbations


def build_noise(ids, K=K, sigma=SIGMA, shape=(3, 32, 32)):
    """Precompute all K perturbations once, seeded per image index (frozen rule). Reused across
    every condition, since the noise depends only on the image and k. Returns CPU (N, K, 3, 32, 32)."""
    out = np.empty((len(ids), K, *shape), dtype=np.float32)
    for i, idx in enumerate(ids):
        for k in range(K):
            rng = np.random.default_rng(int(idx) * 64 + k)
            out[i, k] = rng.standard_normal(shape).astype(np.float32) * sigma
    return torch.from_numpy(out)


def _perturb(x, noise_k):
    return (x + noise_k).clamp(0.0, 1.0)


# ------------------------------------------------------------------ distances between map sets
def _spearman_rows(a, b):
    """1 - Spearman rank correlation, per row. a, b: (N, D) float arrays."""
    ar = a.argsort(1).argsort(1).astype(np.float64)
    br = b.argsort(1).argsort(1).astype(np.float64)
    ar -= ar.mean(1, keepdims=True)
    br -= br.mean(1, keepdims=True)
    num = (ar * br).sum(1)
    den = np.sqrt((ar ** 2).sum(1) * (br ** 2).sum(1))
    rho = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return 1.0 - rho


def _l1_rows(a, b):
    """Normalised L1 between maps scaled to sum 1, per row."""
    a = a / np.clip(a.sum(1, keepdims=True), 1e-12, None)
    b = b / np.clip(b.sum(1, keepdims=True), 1e-12, None)
    return np.abs(a - b).sum(1)


def _cosine_rows(a, b):
    num = (a * b).sum(1)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    cos = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return 1.0 - cos


def map_distances(a0, ak):
    """a0, ak: (N,32,32). Return dict of per-image distances."""
    a = a0.reshape(len(a0), -1)
    b = ak.reshape(len(ak), -1)
    return {"spearman": _spearman_rows(a, b), "l1": _l1_rows(a, b), "cosine": _cosine_rows(a, b)}


# ------------------------------------------------------------------ the two Track-A signals
def operational_drift(model, x, noise, targets, method, K=K, ig_ibs=32):
    """Mean distance between A(x) and A(T_k(x)) over K perturbations. noise: (N,K,3,32,32) on x's device."""
    a0 = compute_maps(method, model, x, targets, ig_ibs=ig_ibs)
    acc = {"spearman": 0.0, "l1": 0.0, "cosine": 0.0}
    for k in range(K):
        xk = _perturb(x, noise[:, k])
        ak = compute_maps(method, model, xk, targets, ig_ibs=ig_ibs)  # SAME target as x
        for key, val in map_distances(a0, ak).items():
            acc[key] = acc[key] + val
    return {f"drift_{method}_{key}": (v / K).astype(np.float32) for key, v in acc.items()}


@torch.no_grad()
def output_consistency(model, x, noise, pred, K=K):
    """Label stability (LS) and confidence deviation (CD) over the same K perturbations."""
    p0_pred = torch.softmax(model(x), dim=1).gather(1, pred.view(-1, 1)).squeeze(1)
    same = torch.zeros(len(x), device=x.device)
    cd = torch.zeros(len(x), device=x.device)
    for k in range(K):
        pk = torch.softmax(model(_perturb(x, noise[:, k])), dim=1)
        same += (pk.argmax(1) == pred).float()
        cd += (p0_pred - pk.gather(1, pred.view(-1, 1)).squeeze(1)).abs()
    return {"out_LS": (same / K).cpu().numpy().astype(np.float32),
            "out_CD": (cd / K).cpu().numpy().astype(np.float32)}


def diagnostic_drift(model, x_clean, x_corr, targets, method, ig_ibs=32):
    """Secondary diagnostic: distance between the clean and corrupted maps of the same image."""
    a_clean = compute_maps(method, model, x_clean, targets, ig_ibs=ig_ibs)
    a_corr = compute_maps(method, model, x_corr, targets, ig_ibs=ig_ibs)
    return {f"diag_{method}_{k}": v.astype(np.float32) for k, v in map_distances(a_clean, a_corr).items()}
