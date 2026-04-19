# Inference

Edit **`inference/inference_config.json`** and run:

```bash
.venv/bin/python inference/run_inference.py
```

Required keys: **`backend`** (basename of a module in **`vlm_models/`** at the repo root, same as training), **`model_name`**, **`image_path`**, **`prompt`**.

Optional: **`bf16`** (`null` = use bfloat16 on CUDA when available) for **`qwen_vl`** only; **`grounding_dino`** always runs in float32 (avoids dtype mismatches in the HF stack). For **generative** backends: **`max_new_tokens`** (default 256). For **Grounding DINO**: **`grounding_threshold`** (default 0.4), **`text_threshold`** (default 0.3), **`grounding_max_detections`** (optional cap); `max_new_tokens` is ignored.

**Backends:** **`qwen_vl`** uses **`load_for_inference`** and chat **`generate`**. **`grounding_dino`** implements **`generate_from_config`** and prints JSON with **`detections`** (label, score, box in pixel xyxy). Use a Grounding DINO Hugging Face id for **`model_name`**: **`IDEA-Research/grounding-dino-base`** is the stronger default (Swin-B class backbone, better zero-shot quality); **`IDEA-Research/grounding-dino-tiny`** matches the repo’s Swin-T OGC demo and is lighter on VRAM. **`prompt`** lists what to detect: one phrase, or several separated by newlines or by **`. `** (period + space).

Training uses **`build_model_collator(cfg)`** where implemented (maze JSONL fine-tuning is **`qwen_vl`** only; **`grounding_dino`** is inference-only here).

## Visualization

For **Grounding DINO** plots, the **`prompt`** from **`inference_config.json`** is drawn under the color legend (word-wrapped). Toggle with **`show_inference_prompt`** and spacing fonts in **`inference_viz/plotting_config.json`**.

Edit **`inference_viz/plotting_config.json`**: set **`enabled`** and **`output_format`**. Destination: non-empty **`save_path`** is a full output file (repo-root-relative or absolute). Otherwise the PNG is ``<image_stem>_inference_viz_<YYYYMMDD_HHMMSS_ffffff>.png`` under **`output_folder`** (repo-root-relative or absolute; default **`temp output`** if the key is omitted). Set **`output_folder`** to JSON **`null`** to save next to the source image instead. **`plot_output_folder`** in **`inference_config.json`** overrides **`output_folder`** from the plotting config (but not **`save_path`**). **`show`**: preview window after saving. New formats: register in **`OUTPUT_FORMAT_PLOTTERS`** in **`inference_viz/inference_visualizer.py`**. From code, **`save_inference_plot(..., output_folder=...)`** overrides inference JSON for that call.
