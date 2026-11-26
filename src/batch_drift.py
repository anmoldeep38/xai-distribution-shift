"""Week 6 — batch-level attribution drift (Track B), frozen protocol.

A batch drift score is shared by a whole evaluation condition; it is NEVER used to rank individual
images (that is Track A). Here we test H2: does aggregate drift track condition-level error?

Per-image profile : |map| summed over channels, average-pooled to an 8x8 grid (frozen), + eps, sum-normalised.
Batch profile P   : mean of the per-image profiles, renormalised (a 64-bin distribution).
Clean reference   : 1000 CIFAR-10 *validation* images (100/class, fixed), profile recomputed per model.
Evaluation batch  : the 2000-image explanation subset under one (corruption, severity) -- disjoint from the reference.
Drift             : Jensen-Shannon distance (base 2) between P_clean and P_condition.
Variability       : 200 bootstrap batches of 500 images per condition -> mean + 95% interval.
Outcome           : error rate of the condition on the full 10,000 corrupted test images.

Output: one row per condition (corruption, severity, js_mean, js_lo, js_hi, error_rate), plus the
Spearman correlation between drift and error rate across conditions (the H2 check).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attributions import compute_maps  # noqa: E402
from data import CORRUPTIONS, N_PER_SEVERITY  # noqa: E402
from models import ResNet18CIFAR  # noqa: E402
from subset import explanation_subset  # noqa: E402

try:
    from data import train_val_split
except Exception:  # pragma: no cover
    train_val_split = None


def load_model(ckpt_path, device):
    model = ResNet18CIFAR().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt.get("args", {}).get("seed", "?")


def profiles(maps, grid):
    """(N,32,32) non-negative maps -> (N, grid*grid) sum-normalised distributions."""
    t = torch.from_numpy(np.ascontiguousarray(maps)).unsqueeze(1).float()  # (N,1,32,32)
    p = F.adaptive_avg_pool2d(t, grid).reshape(len(maps), -1).numpy()
    p = p + 1e-8
    return p / p.sum(1, keepdims=True)


def maps_for(model, x, device, method, batch_size, ig_ibs):
    """Base (unperturbed) maps for the predicted class, batched. Returns (N,32,32)."""
    out = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device)
        with torch.no_grad():
            pred = model(xb).argmax(1)
        out.append(compute_maps(method, model, xb, pred, ig_ibs=ig_ibs))
    return np.concatenate(out, axis=0)


def clean_reference(root, model, device, method, grid, batch_size, ig_ibs, n_per_class=100, seed=0):
    """Mean profile over 1000 clean validation images (100/class, fixed indices)."""
    train = datasets.CIFAR10(root, train=True, download=True, transform=transforms.ToTensor())
    targets = np.array(train.targets)
    if train_val_split is not None:
        _, val_idx = train_val_split(len(targets), seed=0)
    else:
        val_idx = np.arange(len(targets))
    rng = np.random.default_rng(seed)
    val_targets = targets[val_idx]
    chosen = []
    for c in range(10):
        pool = val_idx[val_targets == c]
        chosen.append(rng.choice(pool, size=n_per_class, replace=False))
    chosen = np.sort(np.concatenate(chosen))
    x = torch.stack([train[i][0] for i in chosen])
    p = profiles(maps_for(model, x, device, method, batch_size, ig_ibs), grid)
    ref = p.mean(0) + 1e-8
    return ref / ref.sum()


@torch.no_grad()
def condition_error_rate(c10c_dir, model, device, corruption, severity, batch_size=500):
    """Error rate on the full 10,000 corrupted test images of one condition."""
    arr = np.load(Path(c10c_dir) / f"{corruption}.npy", mmap_mode="r")
    labels = np.load(Path(c10c_dir) / "labels.npy")
    base = (severity - 1) * N_PER_SEVERITY
    wrong = total = 0
    for i in range(0, N_PER_SEVERITY, batch_size):
        sel = np.ascontiguousarray(arr[base + i: base + i + batch_size])
        x = torch.from_numpy(sel).permute(0, 3, 1, 2).float().div_(255.0).to(device)
        pred = model(x).argmax(1).cpu().numpy()
        y = labels[base + i: base + i + batch_size]
        wrong += int((pred != y).sum()); total += len(y)
    return wrong / total


def subset_condition_images(c10c_dir, corruption, severity, idx):
    arr = np.load(Path(c10c_dir) / f"{corruption}.npy", mmap_mode="r")
    sel = np.ascontiguousarray(arr[(severity - 1) * N_PER_SEVERITY + idx])
    return torch.from_numpy(sel).permute(0, 3, 1, 2).float() / 255.0


def bootstrap_js(ref, per_image_profiles, n_boot, boot_size, seed=0):
    """Distribution of JS distance from bootstrap batches of the evaluation set."""
    rng = np.random.default_rng(seed)
    n = len(per_image_profiles)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=min(boot_size, n))
        p = per_image_profiles[pick].mean(0) + 1e-8
        p = p / p.sum()
        vals.append(jensenshannon(ref, p, base=2))
    vals = np.array(vals)
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", choices=["ig", "gradcam"], default="ig",
                   help="protocol default is IG; use gradcam for the early/cheap pass")
    p.add_argument("--c10c-dir", default="data/CIFAR-10-C")
    p.add_argument("--corruption", default="all")
    p.add_argument("--severity", default="all")
    p.add_argument("--grid", type=int, default=8, help="profile grid (sensitivity: 4 or 16)")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--boot-size", type=int, default=500)
    p.add_argument("--data", default="data")
    p.add_argument("--out-dir", default="batch_drift")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ig-ibs", type=int, default=16)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, seed = load_model(args.ckpt, device)
    print(f"device={device} seed={seed} method={args.method} grid={args.grid}")

    test = datasets.CIFAR10(args.data, train=False, download=True)
    idx, _ = explanation_subset(np.array(test.targets))

    ref = clean_reference(args.data, model, device, args.method, args.grid, args.batch_size, args.ig_ibs)
    print(f"clean reference profile built ({args.grid}x{args.grid} = {len(ref)} bins)")

    corruptions = CORRUPTIONS if args.corruption == "all" else [args.corruption]
    severities = [1, 2, 3, 4, 5] if args.severity == "all" else [int(args.severity)]
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    rows = []
    for corr in corruptions:
        for sev in severities:
            x = subset_condition_images(args.c10c_dir, corr, sev, idx)
            pim = profiles(maps_for(model, x, device, args.method, args.batch_size, args.ig_ibs), args.grid)
            js_mean, js_lo, js_hi = bootstrap_js(ref, pim, args.n_boot, args.boot_size)
            err = condition_error_rate(args.c10c_dir, model, device, corr, sev)
            rows.append(dict(corruption=corr, severity=sev, js_mean=round(js_mean, 6),
                             js_lo=round(js_lo, 6), js_hi=round(js_hi, 6), error_rate=round(err, 6)))
            print(f"  {corr:17s} sev {sev}: JS={js_mean:.4f} [{js_lo:.4f},{js_hi:.4f}]  err={err:.3f}")

    fname = out_dir / f"batch_drift_{args.method}_grid{args.grid}_seed{seed}.csv"
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    if len(rows) > 2:
        js = np.array([r["js_mean"] for r in rows]); er = np.array([r["error_rate"] for r in rows])
        rho = spearmanr(js, er).correlation
        print(f"\nH2 check: Spearman(JS drift, error rate) over {len(rows)} conditions = {rho:+.3f}")
    print(f"saved {fname}")


if __name__ == "__main__":
    main()
