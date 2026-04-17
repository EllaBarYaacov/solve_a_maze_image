"""
Generate mazes from ``maze_generation_config.json``: size buckets with target fractions,
``master_seed`` for reproducibility, and per-maze derived seeds so layouts differ even
when side length repeats. Side lengths are always sampled as **odd** integers in each
bucket range (for ``maze_generator_jotbleach``).

Each run writes under ``maze_data/`` into ``maze_data_<YYYYMMDD_HHMMSS_microseconds>/``
(next to ``maze_data_generation/`` by default), and stores a copy of the config file
used for that run at the top of that folder.

  python maze_data_generation/generate_maze_data.py
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from maze import Maze

DEFAULT_CONFIG_PATH = os.path.join(_HERE, "maze_generation_config.json")


def derived_maze_seed(base: int, index: int) -> int:
    """Deterministic per-index seed so same-size mazes still differ in structure."""
    x = (base * 1103515245 + (index + 1) * 12345) & 0x7FFFFFFF
    return x if x != 0 else abs(base) + index + 1


def _allocate_bucket_counts(total: int, fractions: list[float]) -> list[int]:
    """Largest-remainder method so counts sum exactly to ``total``."""
    exact = [total * f for f in fractions]
    base = [math.floor(e) for e in exact]
    rem = total - sum(base)
    frac_parts = sorted(
        ((exact[i] - base[i], i) for i in range(len(base))),
        key=lambda t: t[0],
        reverse=True,
    )
    for k in range(rem):
        base[frac_parts[k][1]] += 1
    return base


def _odd_sizes_in_range(lo: int, hi: int) -> list[int]:
    return [x for x in range(lo, hi + 1) if x % 2 == 1]


def _random_odd_side_in_range(rng: random.Random, lo: int, hi: int) -> int:
    """Sample a random odd side length in ``[lo, hi]`` (inclusive)."""
    if lo > hi:
        raise ValueError(f"Invalid range [{lo}, {hi}]")
    choices = _odd_sizes_in_range(lo, hi)
    if not choices:
        raise ValueError(f"No odd side lengths in [{lo}, {hi}]")
    return rng.choice(choices)


def _validate_config(cfg: dict[str, Any]) -> None:
    if "master_seed" not in cfg:
        raise ValueError("config must include master_seed")
    n = int(cfg["total_mazes"])
    if n < 1:
        raise ValueError("total_mazes must be >= 1")
    buckets = cfg["size_buckets"]
    if not buckets:
        raise ValueError("size_buckets must be non-empty")
    fracs: list[float] = []
    for b in buckets:
        lo, hi = int(b["min_side"]), int(b["max_side"])
        if lo > hi:
            raise ValueError(f"min_side > max_side in bucket {b}")
        f = float(b["fraction"])
        if f <= 0:
            raise ValueError("bucket fraction must be positive")
        fracs.append(f)
    s = sum(fracs)
    if abs(s - 1.0) > 1e-5:
        raise ValueError(f"size_buckets fractions must sum to 1.0, got {s}")


def generate_maze_data(
    config_path: str = DEFAULT_CONFIG_PATH,
    output: str | None = None,
) -> str:
    """
    Load config, sample sizes per bucket, generate ``total_mazes`` maze folders with
    solution sequences + CSVs, and copy ``config_path`` into the batch folder. Returns
    absolute path to the batch folder.
    """
    out = output if output is not None else os.path.join(_REPO_ROOT, "maze_data")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    _validate_config(cfg)

    master = int(cfg["master_seed"])
    total = int(cfg["total_mazes"])
    buckets = cfg["size_buckets"]
    fractions = [float(b["fraction"]) for b in buckets]
    counts = _allocate_bucket_counts(total, fractions)

    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_name = f"maze_data_{now}"
    batch_root = os.path.abspath(os.path.join(out, batch_name))
    os.makedirs(batch_root, exist_ok=True)

    config_copy = os.path.join(batch_root, os.path.basename(config_path))
    shutil.copy2(config_path, config_copy)

    rng = random.Random(master)

    global_idx = 0
    for bi, b in enumerate(buckets):
        lo, hi = int(b["min_side"]), int(b["max_side"])
        for _ in range(counts[bi]):
            side = _random_odd_side_in_range(rng, lo, hi)
            maze_seed = derived_maze_seed(master, global_idx)
            maze = Maze(side, side, maze_seed)
            sub = os.path.join(
                batch_root,
                f"maze_{global_idx:03d}_size_{side}x{side}_seed_{maze_seed}",
            )
            os.makedirs(sub, exist_ok=True)
            maze.create_solution_images_seq(sub, exploratory=False)
            global_idx += 1

    return batch_root


if __name__ == "__main__":
    CONFIG_PATH = DEFAULT_CONFIG_PATH
    OUTPUT = os.path.join(_REPO_ROOT, "maze_data")

    batch_path = generate_maze_data(config_path=CONFIG_PATH, output=OUTPUT)
    print(batch_path)
