"""
``InferenceVisualizer`` and format handlers for drawing model outputs on images.

``visualizer_config.yaml`` in the ``inference_viz`` package directory sets defaults.
Register new output layouts in ``OUTPUT_FORMAT_PLOTTERS``.
"""

from __future__ import annotations

import colorsys
import json
import re

import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

# (image RGB, model_output string, plotting config) -> new RGB image
OutputFormatPlotFn = Callable[[Image.Image, str, dict[str, Any]], Image.Image]


def inference_viz_filename(image_stem: str) -> str:
    """``<stem>_inference_viz_<local time>.png`` (microsecond resolution)."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{image_stem}_inference_viz_{now}.png"


def default_plotting_config_path() -> Path:
    """Path to ``visualizer_config.yaml`` in the ``inference_viz`` package directory."""
    return Path(__file__).resolve().parent / "visualizer_config.yaml"


def load_plotting_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Plotting config must be a YAML mapping: {path}")
    return data


def _hex_rgb(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 6:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    raise ValueError(f"Expected #RRGGBB color, got {s!r}")


def _default_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except OSError:
            return ImageFont.load_default()


def _distinct_box_colors(n: int) -> list[tuple[int, int, int]]:
    """Evenly spaced hues for ``n`` boxes (RGB 0–255)."""
    if n <= 0:
        return []
    colors: list[tuple[int, int, int]] = []
    for i in range(n):
        h = (i + 0.08) / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.82, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _wrap_text_to_pixel_width(
    measure: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_px: int,
) -> list[str]:
    """Greedy word wrap to fit lines under ``max_px`` wide (for PIL drawing)."""
    if not text.strip():
        return []
    if max_px < 12:
        return [text]
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = w if not cur else " ".join(cur + [w])
        b = measure.textbbox((0, 0), trial, font=font)
        if b[2] - b[0] <= max_px:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            b_one = measure.textbbox((0, 0), w, font=font)
            if b_one[2] - b_one[0] <= max_px:
                cur = [w]
            else:
                chunk = ""
                for ch in w:
                    t2 = chunk + ch
                    bt = measure.textbbox((0, 0), t2, font=font)
                    if bt[2] - bt[0] <= max_px:
                        chunk = t2
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                cur = [chunk] if chunk else []
    if cur:
        lines.append(" ".join(cur))
    return lines


def plot_grounding_dino(image: Image.Image, model_output: str, cfg: dict[str, Any]) -> Image.Image:
    """
    Expects JSON like ``{"detections": [{"label": str, "score": float, "box": [x1,y1,x2,y2]}]}``.
    Boxes are xyxy in pixel coordinates. Each box gets a distinct color; class names are in
    the right-hand legend. If ``cfg`` contains ``inference_prompt`` (set by
    ``InferenceVisualizer.render(..., inference_prompt=...)`` from ``inference_config.yaml``),
    that text is drawn below the legend, word-wrapped in the legend column.
    """
    data = json.loads(model_output.strip())
    detections = data.get("detections")
    if not isinstance(detections, list):
        raise ValueError("grounding_dino output must be a JSON object with a 'detections' list")

    rows: list[tuple[tuple[float, float, float, float], str]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        box = det.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        label = det.get("label", "")
        rows.append(((x1, y1, x2, y2), str(label) if label is not None else ""))

    if not rows:
        return image.copy()

    n = len(rows)
    colors = _distinct_box_colors(n)
    iw, ih = image.size
    box_w = int(cfg.get("box_width", 3))
    font_size = int(cfg.get("font_size", 14))
    font = _default_font(font_size)
    legend_pad = int(cfg.get("legend_padding", 12))
    swatch = int(cfg.get("legend_swatch_size", 18))
    gap = int(cfg.get("legend_text_gap", 8))
    row_gap = int(cfg.get("legend_row_gap", 8))
    legend_text_color = _hex_rgb(str(cfg.get("legend_text_color", "#1a1a1a")))
    legend_bg = _hex_rgb(str(cfg.get("legend_bg", "#F0F0F0")))
    divider_color = _hex_rgb(str(cfg.get("legend_divider_color", "#CCCCCC")))

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_tw = 0
    for _, lbl in rows:
        b = measure.textbbox((0, 0), lbl, font=font)
        max_tw = max(max_tw, b[2] - b[0])
    line_h = max(swatch, font_size + 4)
    legend_w = legend_pad + swatch + gap + max_tw + legend_pad
    legend_w = max(legend_w, int(cfg.get("legend_min_width", 120)))

    show_prompt = bool(cfg.get("show_inference_prompt", True))
    prompt_raw = str(cfg.get("inference_prompt", "")).strip() if show_prompt else ""
    prompt_font_size = int(cfg.get("prompt_font_size", max(font_size - 1, 11)))
    font_prompt = _default_font(prompt_font_size)
    prompt_color = _hex_rgb(str(cfg.get("prompt_text_color", "#333333")))
    prompt_section_gap = int(cfg.get("prompt_section_gap", 14))
    prompt_line_gap = int(cfg.get("prompt_line_gap", 5))
    prompt_bottom_pad = int(cfg.get("prompt_bottom_padding", 12))
    max_prompt_px = legend_w - 2 * legend_pad

    prompt_lines: list[str] = []
    prompt_block_h = 0
    if prompt_raw:
        prompt_lines = _wrap_text_to_pixel_width(
            measure, prompt_raw, font_prompt, max_prompt_px
        )
        for pl in prompt_lines:
            b = measure.textbbox((0, 0), pl, font=font_prompt)
            prompt_block_h += (b[3] - b[1]) + prompt_line_gap
        if prompt_lines:
            prompt_block_h -= prompt_line_gap

    last_row_top = legend_pad + (n - 1) * (line_h + row_gap)
    legend_block_bottom = last_row_top + swatch

    prompt_y = legend_block_bottom + prompt_section_gap
    legend_panel_bottom = prompt_y + prompt_block_h + prompt_bottom_pad if prompt_raw else legend_block_bottom + legend_pad
    total_h = max(ih, legend_panel_bottom)

    out = Image.new("RGB", (iw + legend_w, total_h), legend_bg)
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)

    for i, ((x1, y1, x2, y2), _) in enumerate(rows):
        c = colors[i]
        draw.rectangle([x1, y1, x2, y2], outline=c, width=max(1, box_w))

    div_x = iw
    draw.line([(div_x, 0), (div_x, total_h)], fill=divider_color, width=1)

    y = legend_pad
    ty_offset = max(0, (swatch - font_size) // 2)
    for i, (_, lbl) in enumerate(rows):
        c = colors[i]
        sx0 = iw + legend_pad
        sx1 = sx0 + swatch
        sy1 = y + swatch
        draw.rectangle([sx0, y, sx1, sy1], outline=(60, 60, 60), fill=c, width=1)
        draw.text((sx1 + gap, y + ty_offset), lbl, font=font, fill=legend_text_color)
        y += line_h + row_gap

    if prompt_raw and prompt_lines:
        py = float(prompt_y)
        for pl in prompt_lines:
            draw.text((iw + legend_pad, int(py)), pl, font=font_prompt, fill=prompt_color)
            b = measure.textbbox((0, 0), pl, font=font_prompt)
            py += (b[3] - b[1]) + prompt_line_gap

    return out


def plot_qwen_vl_line_json(image: Image.Image, model_output: str, cfg: dict[str, Any]) -> Image.Image:
    """
    Expects a JSON object with ``x1``, ``y1``, ``x2``, ``y2`` (line segment in pixels).
    Optional ``line_width`` for stroke width. Tolerates markdown fences around JSON.
    """
    text = model_output.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    x1, y1 = float(data["x1"]), float(data["y1"])
    x2, y2 = float(data["x2"]), float(data["y2"])
    lw = max(1, int(data.get("line_width", cfg.get("line_width", 4))))

    out = image.copy()
    draw = ImageDraw.Draw(out)
    color = _hex_rgb(str(cfg.get("line_color", "#FF0000")))
    draw.line([(x1, y1), (x2, y2)], fill=color, width=lw)
    return out


OUTPUT_FORMAT_PLOTTERS: dict[str, OutputFormatPlotFn] = {
    "grounding_dino": plot_grounding_dino,
    "qwen_vl_line_json": plot_qwen_vl_line_json,
}


class InferenceVisualizer:
    """Load plotting config (YAML) and render model output onto images."""

    def __init__(self, plotting_config: dict[str, Any]) -> None:
        self.cfg = plotting_config

    def render(
        self,
        image: Image.Image,
        model_output: str,
        output_format: str | None = None,
        *,
        inference_prompt: str | None = None,
    ) -> Image.Image:
        fmt = output_format or str(self.cfg.get("output_format", "")).strip()
        if not fmt:
            raise ValueError(
                "plotting config must set 'output_format' (e.g. 'grounding_dino')"
            )
        plot_fn = OUTPUT_FORMAT_PLOTTERS.get(fmt)
        if plot_fn is None:
            known = ", ".join(sorted(OUTPUT_FORMAT_PLOTTERS))
            raise ValueError(f"Unknown output_format {fmt!r}. Known: {known}")
        cfg = {**self.cfg}
        if inference_prompt is not None:
            cfg["inference_prompt"] = inference_prompt
        return plot_fn(image, model_output, cfg)

    def resolve_save_path(
        self,
        image_path: Path,
        explicit: str | None,
        *,
        relative_base: Path | None = None,
    ) -> Path:
        base = relative_base or image_path.parent
        if explicit:
            p = Path(explicit).expanduser()
            if not p.is_absolute():
                p = (base / p).resolve()
            else:
                p = p.resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        return (image_path.parent / inference_viz_filename(image_path.stem)).resolve()

    def save(
        self,
        image_path: Path | str,
        model_output: str,
        output_format: str | None = None,
        save_path: str | None = None,
        *,
        relative_base: Path | None = None,
        inference_prompt: str | None = None,
    ) -> Path:
        path = Path(image_path)
        image = Image.open(path).convert("RGB")
        rendered = self.render(
            image,
            model_output,
            output_format=output_format,
            inference_prompt=inference_prompt,
        )
        raw_path = save_path if save_path is not None else self.cfg.get("save_path")
        out = self.resolve_save_path(path, raw_path, relative_base=relative_base)
        rendered.save(out)
        if self.cfg.get("show"):
            rendered.show()
        return out
