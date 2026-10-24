"""Score-ordered matching and post-selection suppression."""

from collections import defaultdict

def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = (ax1 + aw, ay1 + ah)
    bx2, by2 = (bx1 + bw, by1 + bh)
    ix1, iy1 = (max(ax1, bx1), max(ay1, by1))
    ix2, iy2 = (min(ax2, bx2), min(ay2, by2))
    iw, ih = (max(0.0, ix2 - ix1), max(0.0, iy2 - iy1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0

def nms(rows, thr=0.5):
    by_img = defaultdict(list)
    for r in rows:
        by_img[int(r['image_id'])].append(r)
    out = []
    for iid, arr in by_img.items():
        arr = sorted(arr, key=lambda r: r['score'], reverse=True)
        kept = []
        for r in arr:
            if all((iou_xywh(r['bbox'], k['bbox']) < thr for k in kept)):
                kept.append(r)
        out.extend(kept)
    return out

def counts_for(rows, gt_by_img, image_ids):
    by_img = defaultdict(list)
    for r in rows:
        by_img[int(r['image_id'])].append(r)
    per_img = {}
    for iid in image_ids:
        preds = sorted(by_img.get(iid, []), key=lambda r: r['score'], reverse=True)
        gts = gt_by_img.get(iid, [])
        matched = set()
        tp = fp = hsfp = 0
        for p in preds:
            best_iou, best_j = (0.0, None)
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                v = iou_xywh(p['bbox'], g['bbox'])
                if v > best_iou:
                    best_iou, best_j = (v, j)
            if best_iou >= 0.5 and best_j is not None:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
                if float(p['score']) >= 0.5:
                    hsfp += 1
        fn = max(0, len(gts) - len(matched))
        per_img[iid] = (tp, fp, fn, hsfp)
    return per_img

def metrics_from_counts(per_img, image_ids):
    tp = fp = fn = hsfp = 0
    for iid in image_ids:
        a, b, c, d = per_img.get(iid, (0, 0, 0, 0))
        tp += a
        fp += b
        fn += c
        hsfp += d
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {'precision': p, 'recall': r, 'f1': f1, 'hs_fp': hsfp, 'tp': tp, 'fp': fp, 'fn': fn}
