"""Week 5 runner: per-image drift + output-consistency features for the subset under given conditions.

Each row of the output is one subset image under one (corruption, severity):
    image_idx, half, corruption, severity, label, pred, correct, conf, entropy,
    out_LS, out_CD, drift_<method>_{spearman,l1,cosine}[, diag_<method>_{...}]

Examples
--------
# cheap: Grad-CAM drift on one corruption, all severities
python src/compute_drift.py --ckpt checkpoints/resnet18_seed0.pt --method gradcam \
    --corruption gaussian_noise --severity all

# the clean reference row set (severity 0)
python src/compute_drift.py --ckpt checkpoints/resnet18_seed0.pt --method gradcam --corruption clean

# everything (heavy -- prefer Colab for IG): all 15 corruptions x 5 severities
python src/compute_drift.py --ckpt checkpoints/resnet18_seed0.pt --method both --corruption all --severity all
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attributions import predict  # noqa: E402
from data import CORRUPTIONS, N_PER_SEVERITY  # noqa: E402
from drift import build_noise, diagnostic_drift, operational_drift, output_consistency  # noqa: E402
from models import ResNet18CIFAR  # noqa: E402
from subset import explanation_subset  # noqa: E402


def load_model(ckpt_path, device):
    model = ResNet18CIFAR().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():       # never trained again -> don't store weight grads (saves VRAM)
        p.requires_grad_(False)
    return model, ckpt.get("args", {}).get("seed", "?")


def clean_subset(root):
    """Return (clean_images float[2000,3,32,32] in [0,1], idx, half, labels)."""
    test = datasets.CIFAR10(root, train=False, download=True, transform=transforms.ToTensor())
    labels_all = np.array(test.targets)
    idx, half = explanation_subset(labels_all)
    imgs = torch.stack([test[i][0] for i in idx])
    return imgs, idx, half, labels_all[idx]


def corrupted_subset(c10c_dir, corruption, severity, idx):
    """Corrupted versions of the subset images at one severity, as float[2000,3,32,32] in [0,1]."""
    arr = np.load(Path(c10c_dir) / f"{corruption}.npy", mmap_mode="r")
    base = (severity - 1) * N_PER_SEVERITY
    sel = np.ascontiguousarray(arr[base + idx])            # (2000,32,32,3) uint8
    return torch.from_numpy(sel).permute(0, 3, 1, 2).float() / 255.0


def process_condition(model, x, clean_x, noise_all, ids, labels, half, corruption, severity,
                      methods, device, args):
    """Compute all per-image features for one condition. Returns list of row dicts."""
    n = len(x)
    bs = args.batch_size
    pred = np.empty(n, np.int64); conf = np.empty(n, np.float32); ent = np.empty(n, np.float32)
    feats = {}
    for i in range(0, n, bs):
        xb = x[i:i + bs].to(device)
        nb = noise_all[i:i + bs].to(device)
        pb, cb, eb = predict(model, xb)
        pred[i:i + bs] = pb.cpu().numpy(); conf[i:i + bs] = cb.cpu().numpy(); ent[i:i + bs] = eb.cpu().numpy()
        batch_feats = {}
        batch_feats.update(output_consistency(model, xb, nb, pb, K=args.K))
        for m in methods:
            batch_feats.update(operational_drift(model, xb, nb, pb, m, K=args.K, ig_ibs=args.ig_ibs))
            if args.diagnostic and clean_x is not None:
                cxb = clean_x[i:i + bs].to(device)
                batch_feats.update(diagnostic_drift(model, cxb, xb, pb, m, ig_ibs=args.ig_ibs))
        for key, val in batch_feats.items():
            feats.setdefault(key, []).append(val)
    cols = {k: np.concatenate(v) for k, v in feats.items()}

    rows = []
    for j in range(n):
        row = dict(image_idx=int(ids[j]), half=half[j], corruption=corruption, severity=severity,
                   label=int(labels[j]), pred=int(pred[j]), correct=int(pred[j] == labels[j]),
                   conf=round(float(conf[j]), 6), entropy=round(float(ent[j]), 6))
        for key in cols:
            row[key] = round(float(cols[key][j]), 6)
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", choices=["ig", "gradcam", "both"], default="gradcam")
    p.add_argument("--c10c-dir", default="data/CIFAR-10-C")
    p.add_argument("--corruption", default="gaussian_noise",
                   help="a corruption name, 'clean', or 'all'")
    p.add_argument("--severity", default="all", help="1-5, or 'all'")
    p.add_argument("--data", default="data")
    p.add_argument("--out-dir", default="drift")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ig-ibs", type=int, default=16)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--diagnostic", action="store_true", help="also compute clean-vs-corrupted drift")
    p.add_argument("--limit", type=int, default=0, help="first N images only (debug)")
    p.add_argument("--overwrite", action="store_true", help="recompute conditions even if the CSV exists")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, seed = load_model(args.ckpt, device)
    methods = ["ig", "gradcam"] if args.method == "both" else [args.method]
    clean_x, idx, half, labels = clean_subset(args.data)
    if args.limit:
        idx, half, labels, clean_x = idx[:args.limit], half[:args.limit], labels[:args.limit], clean_x[:args.limit]
    noise_all = build_noise(idx, K=args.K, sigma=args.sigma)  # once, reused for every condition
    print(f"device={device} seed={seed} methods={methods} subset={len(idx)}")

    corruptions = CORRUPTIONS if args.corruption == "all" else [args.corruption]
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    for corr in corruptions:
        if corr == "clean":
            sev_list = [0]
        else:
            sev_list = [1, 2, 3, 4, 5] if args.severity == "all" else [int(args.severity)]
        for sev in sev_list:
            fname = out_dir / f"drift_{args.method}_{corr}_s{sev}_seed{seed}.csv"
            if fname.exists() and not args.overwrite:          # resume: skip completed conditions
                with open(fname) as f:
                    nrows = sum(1 for _ in f) - 1
                if nrows >= len(idx):
                    print(f"  {corr:17s} sev {sev}: skip ({nrows} rows already)")
                    continue
            x = clean_x if corr == "clean" else corrupted_subset(args.c10c_dir, corr, sev, idx)
            rows = process_condition(model, x, clean_x, noise_all, idx, labels, half, corr, sev,
                                     methods, device, args)
            with open(fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            acc = np.mean([r["correct"] for r in rows])
            print(f"  {corr:17s} sev {sev}: {len(rows)} rows, acc {acc:.3f} -> {fname.name}")
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
