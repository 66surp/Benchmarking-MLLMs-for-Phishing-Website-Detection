
from __future__ import annotations
import os, json
from typing import Any, Dict, List

def load_json_files(data_dir: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    files.sort()
    if max_samples and max_samples > 0:
        files = files[:max_samples]
    samples = []
    for fp in files:
        try:
            with open(fp, 'r', encoding='utf-8') as fr:
                samples.append(json.load(fr))
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
    return samples
