"""
Open-vocabulary detection: Hugging Face Grounding DINO (zero-shot object detection).

Use ``backend``: ``\"grounding_dino\"`` in training or inference JSON configs.
``model_name`` should be a Grounding DINO checkpoint, e.g.
``IDEA-Research/grounding-dino-base`` (recommended accuracy) or ``IDEA-Research/grounding-dino-tiny`` (lighter).

**Inference:** ``prompt`` lists what to detect. Multiple phrases: separate with
newlines, or with ``\". \"`` (period + space) as in the HF docs. Output is JSON
with ``detections`` (label, score, box in xyxy pixel coords).

**Training:** Maze next-step JSONL + ``Trainer`` is only wired for generative
VLMs (``qwen_vl``). This backend raises if you call ``build_model_collator``.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def _parse_grounding_phrases(prompt: str) -> list[str]:
    """Split user prompt into Grounding DINO text queries."""
    p = prompt.strip()
    if not p:
        raise ValueError("prompt must not be empty for Grounding DINO")
    if p.startswith("["):
        try:
            parsed = json.loads(p)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return [x.strip() for x in parsed if x.strip()]
    lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines
    parts = [s.strip() for s in p.split(". ") if s.strip()]
    if len(parts) > 1:
        return parts
    return [p]


def load_for_inference(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """
    Load Grounding DINO + processor (no LoRA).

    Expects ``cfg`` with ``model_name``. Weights and activations use **float32**
    (``bf16`` in config is ignored): half-precision checkpoints mix badly with
    float32 processor outputs in parts of this model and trigger dtype errors.
    """
    model_name = str(cfg["model_name"])
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_name,
        dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, processor


def generate_from_config(cfg: dict[str, Any]) -> str:
    """
    Run zero-shot detection and return a JSON string (detections with boxes).

    Uses ``prompt`` for text queries; optional ``grounding_threshold`` (default
    0.4), ``text_threshold`` (default 0.3), ``grounding_max_detections`` (optional cap).
    """
    model, processor = load_for_inference(cfg)
    image_path = cfg["image_path"]
    pil = Image.open(image_path).convert("RGB")
    phrases = _parse_grounding_phrases(str(cfg["prompt"]))
    text_labels = [phrases]

    inputs = processor(images=pil, text=text_labels, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = inputs.to(dev)

    th = float(cfg.get("grounding_threshold", 0.4))
    text_th = float(cfg.get("text_threshold", 0.3))

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [pil.size[::-1]]
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=th,
        text_threshold=text_th,
        target_sizes=target_sizes,
    )
    result = results[0]
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    max_d = cfg.get("grounding_max_detections")
    detections: list[dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels):
        lab = label if isinstance(label, str) else str(label)
        bx = [round(float(x), 2) for x in box.tolist()]
        detections.append(
            {
                "label": lab,
                "score": round(float(score.item()), 4),
                "box": bx,
            }
        )
    if max_d is not None:
        detections = detections[: int(max_d)]

    return json.dumps({"detections": detections}, indent=2)


def build_model_collator(cfg: dict[str, Any]) -> tuple[Any, Any, Any]:
    raise NotImplementedError(
        "The grounding_dino backend does not support maze next-step JSONL training "
        "with this Trainer (that path is for generative VLMs). Use backend 'qwen_vl' "
        "for LoRA on line JSON, or train Grounding DINO separately with object-detection "
        "labels (COCO-style) and the HF detection APIs."
    )
