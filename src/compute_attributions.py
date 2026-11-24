"""Week 4 runner: compute IG + Grad-CAM heatmaps for the fixed subset, and the sanity check.

Examples
--------
# sanity check first (Adebayo model-randomisation test) -- do this before trusting any map
python src/compute_attributions.py --ckpt checkpoints/resnet18_seed0.pt --sanity

# compute and save clean-image heatmaps for both methods, both splits
python src/compute_attributions.py --ckpt checkpoints/resnet18_seed0.pt --method both --split both

Outputs attributions/attr_<method>_<split>_seed<k>.npz with:
    maps (N,32,32 float16), idx, half, pred, label, correct, conf, entropy
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attributions import compute_maps, predict  # noqa: E402
from models import ResNet18CIFAR  # noqa: E402
from subset import explanation_subset  # noqa: E402


def load_model(ckpt_path, device):
    model = ResNet18CIFAR().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("args", {}).get("seed", "?")


def get_subset_tensors(root, device):
    """Return (x, labels, idx, half) for the fixed 2000-image subset, images in [0,1]."""
    test = datasets.CIFAR10(root, train=False, download=True, transform=transforms.ToTensor())
    labels_all = np.array(test.targets)
    idx, half = explanation_subset(labels_all)
    x = torch.stack([test[i][0] for i in idx]).to(device)   # (2000,3,32,32) in [0,1]
    labels = labels_all[idx]
    return x, labels, idx, half


def run_batched(fn, x, targets, batch_size):
    out = []
    for i in range(0, len(x), batch_size):
        out.append(fn(x[i:i + batch_size], targets[i:i + batch_size]))
    return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------------- sanity check
def _reinit(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.reset_running_stats()


def cascade_steps(model):
    """Yield (name, model) with layers progressively randomised from the logits down (Adebayo et al.)."""
    names = ["fc", "layer4", "layer3", "layer2", "layer1", "stem"]
    rand = copy.deepcopy(model)
    done = []
    for n in names:
        if n == "stem":
            _reinit(rand.conv1)
            _reinit(rand.bn1)
        else:
            _reinit(getattr(rand, n))
        done.append(n)
        yield "+".join(done), copy.deepcopy(rand)


def mean_spearman(ref_maps, other_maps):
    """Mean per-image Spearman rank correlation between two sets of (N,32,32) maps."""
    vals = []
    for a, b in zip(ref_maps, other_maps):
        rho = spearmanr(a.ravel(), b.ravel()).correlation
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def sanity_check(model, x, device, methods, n=64, ig_ibs=32):
    """Model-parameter randomisation test. A faithful map must change as weights are randomised."""
    x = x[:n]
    with torch.no_grad():
        pred = model(x).argmax(1)
    print(f"\nSanity check (Adebayo model-randomisation) on {n} images")
    print("A trustworthy method should fall toward ~0 as more layers are randomised.\n")
    for method in methods:
        ref = compute_maps(method, model, x, pred, ig_ibs=ig_ibs)
        print(f"  {method.upper():8s}  trained->randomised  mean Spearman")
        print(f"    {'(reference)':<28s} 1.000")
        last = None
        for name, rmodel in cascade_steps(model):
            rmodel.to(device).eval()
            rmaps = compute_maps(method, rmodel, x, pred, ig_ibs=ig_ibs)
            last = mean_spearman(ref, rmaps)
            print(f"    {name:<28s} {last:+.3f}")
        verdict = "PASS" if (last is not None and abs(last) < 0.30) else "CHECK"
        print(f"    => fully randomised |rho| {'< 0.30  PASS' if verdict=='PASS' else '>= 0.30  CHECK (map may be model-insensitive)'}\n")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", choices=["ig", "gradcam", "both"], default="both")
    p.add_argument("--split", choices=["F", "H", "both"], default="both")
    p.add_argument("--target", choices=["pred", "true"], default="pred",
                   help="class to explain; 'pred' is the operational setting, 'true' is a diagnostic")
    p.add_argument("--data", default="data")
    p.add_argument("--out-dir", default="attributions")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ig-ibs", type=int, default=32, help="IG internal batch size (lower if OOM)")
    p.add_argument("--limit", type=int, default=0, help="only process the first N images (debug)")
    p.add_argument("--sanity", action="store_true", help="run the randomisation sanity check and exit")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, seed = load_model(args.ckpt, device)
    methods = ["ig", "gradcam"] if args.method == "both" else [args.method]
    print(f"device={device}  checkpoint seed={seed}  methods={methods}")

    x, labels, idx, half = get_subset_tensors(args.data, device)
    pred, conf, entropy = predict(model, x)
    pred = pred.cpu().numpy(); conf = conf.cpu().numpy(); entropy = entropy.cpu().numpy()
    print(f"subset={len(idx)}  clean subset accuracy={np.mean(pred == labels):.4f}")

    if args.sanity:
        sanity_check(model, x, device, methods, ig_ibs=args.ig_ibs)
        return

    target_np = pred if args.target == "pred" else labels
    targets = torch.as_tensor(target_np, device=device)
    splits = ["F", "H"] if args.split == "both" else [args.split]
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    for method in methods:
        maps = run_batched(
            lambda xb, tb: compute_maps(method, model, xb, tb, ig_ibs=args.ig_ibs),
            x if not args.limit else x[:args.limit],
            targets if not args.limit else targets[:args.limit],
            args.batch_size,
        )
        n = len(maps)
        for s in splits:
            sel = (half[:n] == s)
            fname = out_dir / f"attr_{method}_{s}_seed{seed}.npz"
            np.savez_compressed(
                fname,
                maps=maps[sel].astype(np.float16),
                idx=idx[:n][sel], half=half[:n][sel],
                pred=pred[:n][sel], label=labels[:n][sel],
                correct=(pred[:n] == labels[:n])[sel],
                conf=conf[:n][sel], entropy=entropy[:n][sel],
                method=method, target=args.target, seed=str(seed),
            )
            print(f"saved {fname}  ({sel.sum()} maps, shape {maps.shape[1:]})")


if __name__ == "__main__":
    main()
