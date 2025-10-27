
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import importlib
from PIL import Image
from ..prompts import ICR_SYSTEM, ICR_USER_TEMPLATE, build_inputs_block
from ..util import balanced_json_extract

@dataclass
class Evidence:
    url_spans: List[str]
    dom_selectors: List[str]
    image_boxes: List[List[float]]

@dataclass
class ModelOutput:
    label: str
    confidence: float
    evidence: Evidence
    rationale: str

class BaseMLLM:
    def __init__(self, model_id: str, device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._lazy_init()

    def _lazy_init(self):
        raise NotImplementedError

    def predict(self, url: Optional[str], html: Optional[str], image_path: Optional[str]) -> ModelOutput:
        raise NotImplementedError

    @staticmethod
    def _postprocess(obj: Dict[str, Any]) -> ModelOutput:
        label = str(obj.get("label", "")).strip().lower()
        if label not in {"phishing", "legit"}:
            label = "abstain"
        conf = float(obj.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        ev = obj.get("evidence", {}) or {}
        url_spans = list({s.strip() for s in (ev.get("url_spans") or []) if isinstance(s, str) and s.strip()})
        dom_selectors = list({s.strip() for s in (ev.get("dom_selectors") or []) if isinstance(s, str) and s.strip()})
        image_boxes = []
        for bb in (ev.get("image_boxes") or []):
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                x1,y1,x2,y2 = [float(v) for v in bb]
                if x2 > x1 and y2 > y1:
                    image_boxes.append([x1,y1,x2,y2])
        rationale = str(obj.get("rationale", "")).strip()
        return ModelOutput(label=label, confidence=conf, evidence=Evidence(url_spans, dom_selectors, image_boxes), rationale=rationale)

def make_adapter(name: str) -> BaseMLLM:
    key = name.strip().lower()
    if key in {"qwen2-vl-7b-instruct", "qwen2-vl-7b", "qwen-vl-7b"}:
        from .qwen2_vl import Qwen2VLAdapter
        return Qwen2VLAdapter("Qwen/Qwen2-VL-7B-Instruct")
    if key in {"qwen2-vl-3b-instruct", "qwen2.5-vl-3b", "qwen2.5-3b"}:
        from .qwen2_vl import Qwen2VLAdapter
        return Qwen2VLAdapter("Qwen/Qwen2-VL-2B-Instruct")
    if key in {"llava-onevision-7b", "llava-ov-7b", "llava-8b"}:
        from .llava_ov import LLaVAOneVisionAdapter
        return LLaVAOneVisionAdapter("llava-hf/LLaVA-OneVision-Qwen2-7B-OV")
    if key in {"phi-3.5-vision-instruct", "phi-3.5-vision", "phi3.5-vision"}:
        from .phi3_vision import Phi35VisionAdapter
        return Phi35VisionAdapter("microsoft/Phi-3.5-vision-instruct")
    raise ValueError(f"Unknown model alias: {name}")
