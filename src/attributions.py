"""Attribution (heatmap) methods, frozen to the protocol.

Integrated Gradients (primary): Captum, black-image baseline (zeros in [0,1] space), 32 steps,
    Gauss-Legendre. Map = |attribution| summed over colour channels -> 32x32, non-negative.
Grad-CAM (secondary, CNN only): weights = global-average-pooled gradients of the predicted-class
    logit w.r.t. the layer4 output (4x4); ReLU of the weighted sum; bilinear upsample to 32x32.

Both return float32 arrays of shape (N, 32, 32), non-negative, one map per image, explaining the
given target class (predicted class at deployment; ground-truth class only as a diagnostic).

The model takes images in [0,1] and normalises internally, so the IG baseline (a black image) and
the inputs live in the same raw-pixel space.
"""
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

IG_STEPS = 32
IG_METHOD = "gausslegendre"


@torch.no_grad()
def predict(model, x):
    """Return (pred, confidence, entropy) for a batch already on the model's device."""
    logits = model(x)
    p = torch.softmax(logits, dim=1)
    conf, pred = p.max(dim=1)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
    return pred, conf, entropy


def ig_maps(model, x, targets, steps=IG_STEPS, internal_batch_size=32):
    """Integrated Gradients maps, shape (N, 32, 32) float32."""
    model.eval()
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x)  # black image in [0,1] space
    att = ig.attribute(x, baselines=baseline, target=targets, n_steps=steps,
                       method=IG_METHOD, internal_batch_size=internal_batch_size)
    m = att.abs().sum(dim=1)  # sum over colour channels -> (N, 32, 32)
    return m.detach().float().cpu().numpy()


def gradcam_maps(model, x, targets):
    """Grad-CAM maps on layer4, shape (N, 32, 32) float32."""
    model.eval()
    store = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    try:
        x = x.detach().clone().requires_grad_(True)  # graph via input; model params stay frozen
        logits = model(x)
        act = store["act"]                                   # (N, 512, 4, 4), requires grad
        selected = logits.gather(1, targets.view(-1, 1)).sum()
        # autograd.grad computes the gradient and frees the graph immediately; it stores no .grad,
        # so nothing accumulates across calls (the .backward()+hook version leaked here).
        grad = torch.autograd.grad(selected, act)[0].detach()
        act = act.detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)        # (N, 512, 1, 1)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))  # (N, 1, 4, 4)
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
        return cam.squeeze(1).float().cpu().numpy()          # (N, 32, 32)
    finally:
        h.remove()
        store.clear()


def compute_maps(method, model, x, targets, ig_steps=IG_STEPS, ig_ibs=32):
    if method == "ig":
        return ig_maps(model, x, targets, steps=ig_steps, internal_batch_size=ig_ibs)
    if method == "gradcam":
        return gradcam_maps(model, x, targets)
    raise ValueError(f"unknown method {method!r} (use 'ig' or 'gradcam')")
