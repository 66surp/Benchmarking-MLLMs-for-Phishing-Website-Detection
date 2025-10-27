
# Phish MLLM Benchmark

A reproducible **phishing website detection** benchmark skeleton for open multimodal LLMs (Qwen2‑VL, LLaVA‑OneVision, Phi‑3.5‑Vision).

## Features
- Unified dataset schema (URL / HTML / screenshot).
- ICR prompting with strict-JSON output (label, confidence, evidence, rationale).
- Adapters for popular open MLLMs via HuggingFace Transformers.
- Evidence scoring (URL substring / DOM selector / image IoU≥0.5).
- Classification metrics (Acc/Prec/Recall/F1).
- McNemar test + Benjamini–Hochberg FDR correction.
- CLI + example dataset; GitHub-ready packaging.

## Install
```bash
pip install --upgrade pip
pip install -r requirements.txt
# Install torch per your CUDA version, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quickstart
```bash
python -m phishbench.cli   --data_dir examples/sample_dataset   --models Qwen2-VL-7B-Instruct,LLaVA-OneVision-7B,phi-3.5-vision-instruct   --modalities url,html,image,all   --max_samples 0   --out_dir ./benchmark_out
```

Or after editable install:
```bash
pip install -e .
phishbench --data_dir examples/sample_dataset --models Qwen2-VL-7B-Instruct --modalities all
```

## Dataset schema
See `examples/sample_dataset/*.json` for a minimal format.

## Notes
- Checkpoint IDs are examples; swap any compatible HF models.
- If a model returns malformed JSON, the parser tries to recover and you still get a record for debugging.
- For long runs, consider chunking data and running per‑modality to control VRAM.

## License
MIT
