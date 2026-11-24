"""Week 1 baseline: ResNet-18 on CIFAR-10 (45k train / 5k val / 10k test).

Usage:
    python src/train_resnet18.py --epochs 40 --seed 0
Writes checkpoints/resnet18_seed{seed}.pt and runs/resnet18_seed{seed}.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import cifar10_loaders  # noqa: E402
from models import ResNet18CIFAR  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total, loss_sum / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0, help="training seed (init, shuffling, augmentation)")
    p.add_argument("--split-seed", type=int, default=0, help="train/val split seed; keep fixed across runs")
    p.add_argument("--data", default="data")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}")
    if device == "cpu":
        print("WARNING: no GPU found; this will be very slow. Use Colab (T4) or a CUDA build of PyTorch.")

    loaders, split = cifar10_loaders(args.data, args.batch_size, args.split_seed, args.workers)
    print(f"train={len(split['train'])} val={len(split['val'])} test={len(loaders['test'].dataset)}")

    model = ResNet18CIFAR().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loaders["train"]), pct_start=0.15
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    history, best_val, t0 = [], 0.0, time.time()
    Path("checkpoints").mkdir(exist_ok=True)
    Path("runs").mkdir(exist_ok=True)
    ckpt_path = Path("checkpoints") / f"resnet18_seed{args.seed}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = run_correct = run_n = 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * y.numel()
            run_correct += (logits.argmax(1) == y).sum().item()
            run_n += y.numel()

        val_acc, val_loss = evaluate(model, loaders["val"], device)
        row = dict(epoch=epoch, train_loss=run_loss / run_n, train_acc=run_correct / run_n,
                   val_loss=val_loss, val_acc=val_acc, minutes=(time.time() - t0) / 60)
        history.append(row)
        print(f"[{epoch:3d}/{args.epochs}] loss {row['train_loss']:.3f} train {row['train_acc']:.4f} "
              f"val {val_acc:.4f} ({row['minutes']:.1f} min)")
        # Model selection uses validation only; the test set is touched once, at the end.
        if val_acc >= best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args),
                        "split": {k: v.tolist() for k, v in split.items()}}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_acc, test_loss = evaluate(model, loaders["test"], device)
    summary = dict(args=vars(args), best_epoch=ckpt["epoch"], best_val_acc=best_val,
                   test_acc=test_acc, test_loss=test_loss, history=history,
                   torch=torch.__version__, device=torch.cuda.get_device_name(0) if use_amp else "cpu")
    with open(Path("runs") / f"resnet18_seed{args.seed}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nbest epoch {ckpt['epoch']}  val {best_val:.4f}  CLEAN TEST ACC {test_acc:.4f}")
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
