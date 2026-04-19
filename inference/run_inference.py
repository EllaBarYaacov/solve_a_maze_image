"""
Run VLM generation from ``inference_config.json`` (backend + model + image + prompt).

Edit the config file; do not pass model settings on the command line.
Generative backends (e.g. ``qwen_vl``) use ``load_for_inference`` + chat generate.
Detection backends (e.g. ``grounding_dino``) implement ``generate_from_config(cfg)``
instead and return structured text (e.g. JSON).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference_viz import (
    InferenceVisualizer,
    default_plotting_config_path,
    inference_viz_filename,
    load_plotting_config,
)
from vlm_models import load_backend_module


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    required = ("backend", "image_path", "prompt", "model_name")
    for k in required:
        if k not in cfg:
            raise KeyError(f"Config must include {k!r}: {config_path}")
    return cfg


def resolve_image_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


_DEFAULT_VIZ_DIR = "temp output"


def _inference_viz_save_path(
    pcfg: dict,
    inference_cfg: dict,
    image_path: Path,
    output_folder: Path | str | None,
) -> str:
    """
    Absolute path for ``<stem>_inference_viz_<timestamp>.png``.

    Precedence: non-empty ``save_path`` in plotting config (file path) >
    ``output_folder`` argument > ``plot_output_folder`` in inference config >
    ``output_folder`` in plotting config. If that key is missing, use
    ``{_DEFAULT_VIZ_DIR!r}`` under the repo root. If plotting config sets
    ``output_folder`` to JSON null, save next to the source image.
    """
    raw_save = pcfg.get("save_path")
    if isinstance(raw_save, str) and raw_save.strip():
        p = Path(raw_save.strip()).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        else:
            p = p.resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    folder_raw: Path | str | None
    folder_raw = output_folder
    if folder_raw is None:
        folder_raw = inference_cfg.get("plot_output_folder")
    if folder_raw is None:
        if "output_folder" in pcfg:
            folder_raw = pcfg["output_folder"]
        else:
            folder_raw = _DEFAULT_VIZ_DIR

    if folder_raw is None:
        out = (image_path.parent / inference_viz_filename(image_path.stem)).resolve()
        return str(out)

    folder = Path(folder_raw).expanduser()
    if not folder.is_absolute():
        folder = (_REPO_ROOT / folder).resolve()
    else:
        folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / inference_viz_filename(image_path.stem))


def run_inference(cfg: dict) -> str:
    image_path = resolve_image_path(str(cfg["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"image_path not found: {image_path}")

    prompt = str(cfg["prompt"])
    backend_name = str(cfg["backend"])
    backend = load_backend_module(backend_name)
    cfg_infer = {**cfg, "image_path": str(image_path)}

    if hasattr(backend, "generate_from_config"):
        return backend.generate_from_config(cfg_infer)

    if not hasattr(backend, "load_for_inference"):
        raise AttributeError(
            f"Backend {backend_name!r} must define load_for_inference(cfg) "
            "or generate_from_config(cfg)"
        )

    model, processor = backend.load_for_inference(cfg_infer)

    pil = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    dev = next(model.parameters()).device
    inputs = {
        k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    max_new = int(cfg.get("max_new_tokens", 256))

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id,
        )

    in_len = inputs["input_ids"].shape[1]
    new_tokens = gen[0, in_len:]
    text = tok.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return text


def save_inference_plot(
    inference_cfg: dict,
    image_path: Path,
    model_output: str,
    output_folder: Path | str | None = None,
    *,
    plotting_config_path: Path | None = None,
) -> Path | None:
    """
    If ``inference_viz/plotting_config.json`` exists and ``enabled`` is true, draw and save a viz.

    ``output_folder`` overrides the destination directory (see
    ``_inference_viz_save_path``). By default, PNGs go under ``temp output/`` at
    the repo root unless that config sets ``output_folder`` (use JSON ``null`` to
    save next to the source image) or ``save_path`` to a file path.
    """
    pcp = plotting_config_path or default_plotting_config_path()
    if not pcp.is_file():
        return None
    try:
        pcfg = load_plotting_config(pcp)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not load plotting config {pcp}: {e}", file=sys.stderr)
        return None
    if not pcfg.get("enabled", False):
        return None
    fmt = pcfg.get("output_format") or inference_cfg.get("backend")
    if not fmt:
        print("Warning: plotting enabled but output_format is missing", file=sys.stderr)
        return None
    save_path = _inference_viz_save_path(pcfg, inference_cfg, image_path, output_folder)
    try:
        visualizer = InferenceVisualizer(pcfg)
        out_path = visualizer.save(
            image_path,
            model_output,
            output_format=str(fmt),
            save_path=save_path,
            relative_base=_REPO_ROOT,
            inference_prompt=inference_cfg.get("prompt"),
        )
    except Exception as e:
        print(f"Warning: inference plot failed: {e}", file=sys.stderr)
        return None
    print(f"Saved visualization: {out_path}")
    return out_path


if __name__ == "__main__":
    CONFIG_PATH = Path(__file__).resolve().parent / "inference_config.json"
    config = load_config(CONFIG_PATH)
    print(f"Config: {CONFIG_PATH}")
    print(f"  backend: {config['backend']}")
    image_resolved = resolve_image_path(str(config["image_path"]))
    print(f"  image_path: {image_resolved}")
    print(f"  model_name: {config['model_name']}")
    out = run_inference(config)
    print("--- model output ---")
    print(out)
    plot_dir = "plot_output_folder"
    save_inference_plot(config, image_resolved, out, plot_dir)
