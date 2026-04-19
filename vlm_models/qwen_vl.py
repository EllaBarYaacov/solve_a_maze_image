"""
Vision–language backend: Hugging Face Qwen3-VL Instruct + LoRA (train) + generate (infer).

Select this backend by setting ``"backend": "qwen_vl"`` in training or inference JSON config.
"""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class MazeVlmCollator:
    """Batch size 1: image + instruction + target JSON; masks loss to assistant tokens."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.pad_id = processor.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError(
                "This collator supports batch size 1 only; increase "
                "gradient_accumulation_steps instead."
            )
        f = features[0]
        image = Image.open(f["image_path"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f["instruction"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f["target"]}],
            },
        ]
        full = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        user_only = self.processor.apply_chat_template(
            [messages[0]],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        nu = int(user_only["input_ids"].shape[1])
        labels = full["input_ids"].clone()
        labels[:, :nu] = -100
        labels[labels == self.pad_id] = -100
        full["labels"] = labels
        return full


def _resolve_bf16(cfg: dict[str, Any]) -> bool:
    use_bf16 = cfg.get("bf16")
    if use_bf16 is None:
        return torch.cuda.is_available()
    return bool(use_bf16)


def load_for_inference(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """
    Load base model + processor for text generation (no LoRA).

    Expects ``cfg`` with ``model_name``; optional ``bf16`` (default: on if CUDA).
    """
    model_name = str(cfg["model_name"])
    use_bf16 = _resolve_bf16(cfg)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def build_model_collator(cfg: dict[str, Any]) -> tuple[Any, Any, MazeVlmCollator]:
    """
    Load processor and causal LM, attach LoRA, return ``(model, processor, collator)``.

    Expects ``cfg`` to include at least: ``model_name``, ``lora_r``, ``lora_alpha``,
    ``lora_dropout``, ``bf16`` (optional bool), ``gradient_checkpointing`` (bool).
    """
    model_name = str(cfg["model_name"])
    use_bf16 = _resolve_bf16(cfg)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    collator = MazeVlmCollator(processor)
    return model, processor, collator
