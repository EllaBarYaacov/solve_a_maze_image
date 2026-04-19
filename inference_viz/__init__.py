"""
Render and save visualizations of model inference (boxes, lines, legends).

Public API is defined in ``inference_visualizer.py``; this file re-exports it so you
can ``import inference_viz`` or ``from inference_viz import InferenceVisualizer``.
"""

from __future__ import annotations

from .inference_visualizer import (
    OUTPUT_FORMAT_PLOTTERS,
    InferenceVisualizer,
    default_plotting_config_path,
    inference_viz_filename,
    load_plotting_config,
)

__all__ = [
    "OUTPUT_FORMAT_PLOTTERS",
    "InferenceVisualizer",
    "default_plotting_config_path",
    "inference_viz_filename",
    "load_plotting_config",
]
