"""Frozen box perturbation and horizontal-flip comparison operations."""

import numpy as np
import torch

EPS = 1e-9

@torch.no_grad()
def refine_boxes(model, img_tensor, boxes_xyxy):
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), np.float32)
    dev = next(model.parameters()).device
    images, _ = model.transform([img_tensor.to(dev)], None)
    feats = model.backbone(images.tensors)
    orig = (img_tensor.shape[-2], img_tensor.shape[-1])
    new = images.image_sizes[0]
    sy, sx = (new[0] / orig[0], new[1] / orig[1])
    b = torch.as_tensor(np.asarray(boxes_xyxy), dtype=torch.float32, device=dev).clone()
    b[:, [0, 2]] *= sx
    b[:, [1, 3]] *= sy
    pooled = model.roi_heads.box_roi_pool(feats, [b], images.image_sizes)
    bf = model.roi_heads.box_head(pooled)
    cls_logits, box_deltas = model.roi_heads.box_predictor(bf)
    ncls = cls_logits.shape[1]
    deltas_obj = box_deltas.reshape(-1, ncls, 4)[:, 1, :]
    refined = model.roi_heads.box_coder.decode(deltas_obj.reshape(-1, 4), [b]).reshape(-1, 4)
    refined[:, [0, 2]] /= sx
    refined[:, [1, 3]] /= sy
    return refined.detach().cpu().numpy().astype(np.float32)

def jitter_variance_for_boxes(model, img_tensor, boxes_xyxy, n_jitter, frac, rng):
    n = len(boxes_xyxy)
    if n == 0:
        return np.zeros(0, np.float32)
    jit = np.empty((n * n_jitter, 4), np.float32)
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        for j in range(n_jitter):
            dx = rng.uniform(-frac, frac) * w
            dy = rng.uniform(-frac, frac) * h
            dw = rng.uniform(-frac, frac) * w
            dh = rng.uniform(-frac, frac) * h
            jit[i * n_jitter + j] = [x1 + dx, y1 + dy, x2 + dw, y2 + dh]
    refined = refine_boxes(model, img_tensor, jit)
    refined = refined.reshape(n, n_jitter, 4)
    out = np.empty(n, np.float32)
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        norm = 0.5 * (h + w) + 1e-06
        sig = refined[i].std(axis=0)
        out[i] = float(sig.mean() / norm)
    return out

def _iou_matrix(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), np.float32)
    ax1, ay1, ax2, ay2 = (A[:, 0][:, None], A[:, 1][:, None], A[:, 2][:, None], A[:, 3][:, None])
    bx1, by1, bx2, by2 = (B[:, 0][None, :], B[:, 1][None, :], B[:, 2][None, :], B[:, 3][None, :])
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aarea = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    barea = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = aarea + barea - inter + EPS
    return (inter / union).astype(np.float32)
