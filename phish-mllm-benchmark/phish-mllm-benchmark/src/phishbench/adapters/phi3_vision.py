
from __future__ import annotations
from typing import Optional
import importlib
from PIL import Image
from .base import BaseMLLM, ModelOutput
from ..prompts import ICR_SYSTEM, ICR_USER_TEMPLATE, build_inputs_block
from ..util import balanced_json_extract

class Phi35VisionAdapter(BaseMLLM):
    def _lazy_init(self):
        transformers = importlib.import_module('transformers')
        torch = importlib.import_module('torch')
        self.processor = transformers.AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=getattr(torch, 'bfloat16'), device_map='auto', trust_remote_code=True
        )

    def predict(self, url: Optional[str], html: Optional[str], image_path: Optional[str]) -> ModelOutput:
        torch = importlib.import_module('torch')
        transformers = importlib.import_module('transformers')
        inputs_block = build_inputs_block(url, html, image_path)
        user = ICR_USER_TEMPLATE.format(inputs_block=inputs_block)

        content = []
        image_obj = None
        if image_path:
            image_obj = Image.open(image_path).convert("RGB")
            content.append({"type": "image", "image": image_obj})
        content.append({"type": "text", "text": user})

        messages = [{"role": "system", "content": ICR_SYSTEM},
                    {"role": "user", "content": content}]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=[image_obj] if image_obj else None, text=prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        obj = balanced_json_extract(text)
        return self._postprocess(obj)
