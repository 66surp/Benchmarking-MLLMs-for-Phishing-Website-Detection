
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from .util import iou_xyxy

@dataclass
class Evidence:
    url_spans: List[str]
    dom_selectors: List[str]
    image_boxes: List[List[float]]

def classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    tp = sum((yt == 'phishing' and yp == 'phishing') for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 'legit' and yp == 'legit') for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 'legit' and yp == 'phishing') for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 'phishing' and yp == 'legit') for yt, yp in zip(y_true, y_pred))
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def evidence_scores(pred: Evidence, gt: Dict[str, Any], iou_thresh: float = 0.5) -> Tuple[int, int, int]:
    gt_url = [str(x).lower() for x in (gt.get('url_indicators') or [])]
    pr_url = [str(x).lower() for x in (pred.url_spans or [])]
    url_hit = any(any((s in g) or (g in s) for g in gt_url) for s in pr_url)

    gt_dom = [str(x).lower() for x in (gt.get('dom_indicators') or [])]
    pr_dom = [str(x).lower() for x in (pred.dom_selectors or [])]
    dom_hit = any(any((s in g) or (g in s) for g in gt_dom) for s in pr_dom)

    gt_boxes = [list(map(float, bb)) for bb in (gt.get('image_indicators') or []) if isinstance(bb, (list, tuple)) and len(bb) == 4]
    pr_boxes = [list(map(float, bb)) for bb in (pred.image_boxes or []) if isinstance(bb, (list, tuple)) and len(bb) == 4]
    img_hit = False
    if gt_boxes and pr_boxes:
        for pb in pr_boxes:
            if any(iou_xyxy(pb, gb) >= iou_thresh for gb in gt_boxes):
                img_hit = True
                break

    hits = sum([url_hit, dom_hit, img_hit])
    target = sum([1 if gt_url else 0, 1 if gt_dom else 0, 1 if gt_boxes else 0])
    tp = hits
    fp = max(0, (len([x for x in [pr_url, pr_dom, pr_boxes] if x]) - hits))
    fn = max(0, target - hits)
    return tp, fp, fn
