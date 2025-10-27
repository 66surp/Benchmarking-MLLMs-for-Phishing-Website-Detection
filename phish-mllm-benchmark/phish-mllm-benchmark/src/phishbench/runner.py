
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
from .data import load_json_files
from .metrics import classification_metrics, evidence_scores, Evidence
from .stats import mcnemar_test, bh_correction
from .util import ensure_dir
from .adapters.base import BaseMLLM, make_adapter

@dataclass
class ModelOutput:
    label: str
    confidence: float
    evidence: Evidence
    rationale: str

def run_model_on_samples(adapter: BaseMLLM, samples: List[Dict[str, Any]], modality: str) -> Tuple[List[str], List[ModelOutput]]:
    y_pred, outputs = [], []
    for ex in tqdm(samples, desc=f"{adapter.model_id} [{modality}]"):
        url = ex.get('inputs', {}).get('url') if modality in {"url", "all"} else None
        html = ex.get('inputs', {}).get('html') if modality in {"html", "all"} else None
        img_path = ex.get('inputs', {}).get('image_path') if modality in {"image", "all"} else None
        try:
            out = adapter.predict(url, html, img_path)
        except Exception as e:
            out = ModelOutput(label="abstain", confidence=0.0, evidence=Evidence([],[],[]), rationale=f"error: {e}")
        y_pred.append(out.label)
        outputs.append(out)
    return y_pred, outputs

def evaluate(samples: List[Dict[str, Any]], y_pred: List[str], outputs: List[ModelOutput]) -> Dict[str, Any]:
    y_true = [str(s['label']).lower() for s in samples]
    y_pred_eval = [p if p in {"phishing","legit"} else ("phishing" if t=="legit" else "legit") for p,t in zip(y_pred, y_true)]
    cls = classification_metrics(y_true, y_pred_eval)

    tp = fp = fn = 0
    for s, out in zip(samples, outputs):
        gt = s.get('annotated_evidence') or {}
        tpi, fpi, fni = evidence_scores(out.evidence, gt)
        tp += tpi; fp += fpi; fn += fni
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    return {"classification": cls, "evidence": {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}}

def run_benchmark(data_dir: str, model_aliases: List[str], modalities: List[str], max_samples: int, out_dir: str):
    ensure_dir(out_dir)
    samples = load_json_files(data_dir, max_samples)
    if not samples:
        raise RuntimeError("No samples loaded. Check --data_dir.")

    y_true = [str(s['label']).lower() for s in samples]

    for modality in modalities:
        all_preds: Dict[str, List[str]] = {}
        summary_rows = []

        for alias in model_aliases:
            adapter = make_adapter(alias)
            y_pred, outputs = run_model_on_samples(adapter, samples, modality)
            all_preds[alias] = y_pred
            metrics = evaluate(samples, y_pred, outputs)

            out_path = os.path.join(out_dir, f"outputs_{alias}_{modality}.jsonl")
            with open(out_path, 'w', encoding='utf-8') as fw:
                for ex, out in zip(samples, outputs):
                    rec = {"id": ex.get('id'), "gt": ex.get('label'),
                           "pred": {"label": out.label, "confidence": out.confidence,
                                    "evidence": {"url_spans": out.evidence.url_spans,
                                                 "dom_selectors": out.evidence.dom_selectors,
                                                 "image_boxes": out.evidence.image_boxes},
                                    "rationale": out.rationale}}
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

            cls = metrics['classification']; ev = metrics['evidence']
            summary_rows.append({'model': alias, 'modality': modality,
                                 'acc': cls['accuracy'], 'prec': cls['precision'], 'rec': cls['recall'], 'f1': cls['f1'],
                                 'ev_prec': ev['precision'], 'ev_rec': ev['recall'], 'ev_f1': ev['f1']})

        import pandas as pd
        import itertools
        df_sum = pd.DataFrame(summary_rows)
        sum_csv = os.path.join(out_dir, f"summary_{modality}.csv"); df_sum.to_csv(sum_csv, index=False)

        rows = []
        names = list(all_preds.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                p, n01, n10 = mcnemar_test(y_true, all_preds[a], all_preds[b])
                rows.append({"model_a": a, "model_b": b, "p_raw": p, "n01": n01, "n10": n10})
        df_sig = pd.DataFrame(rows)
        if not df_sig.empty:
            df_sig["p_adj_bh"] = bh_correction(df_sig["p_raw"].tolist())
        sig_csv = os.path.join(out_dir, f"mcnemar_{modality}.csv"); df_sig.to_csv(sig_csv, index=False)
