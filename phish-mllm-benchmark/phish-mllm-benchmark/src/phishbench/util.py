
from __future__ import annotations
import os, json, re
from typing import Any, Dict, List

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-8
    return float(inter / union)

def balanced_json_extract(text: str) -> Dict[str, Any]:
    # Prefer fenced code blocks
    m = re.findall(r"```json(.*?)```", text, flags=re.S)
    candidate = m[-1] if m else text
    # Try from end to find a balanced {...}
    s = candidate
    last_close = s.rfind('}')
    for end in range(last_close, -1, -1):
        if s[end] != '}':
            continue
        depth = 0
        for start in range(end, -1, -1):
            ch = s[start]
            if ch == '}':
                depth += 1
            elif ch == '{':
                depth -= 1
                if depth == 0:
                    block = s[start:end+1]
                    # Try json, then json5
                    try:
                        return json.loads(block)
                    except Exception:
                        try:
                            import json5
                            return json5.loads(block)
                        except Exception:
                            break
        # continue scanning earlier closings
    # Fallback: any braces
    mm = re.search(r"\{[\s\S]*\}", text)
    if mm:
        block = mm.group(0)
        try:
            return json.loads(block)
        except Exception:
            import json5
            return json5.loads(block)
    raise ValueError("No JSON object found")
