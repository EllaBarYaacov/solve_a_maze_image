"""Maze grid with labeled cells, PNG export, and round-trip image loading."""

from __future__ import annotations

from collections import deque
import os
import random
import re
from typing import ClassVar

import numpy as np
from maze_generator_jotbleach import MazeGenerator
from PIL import Image, ImageDraw

# Default RGB (distinct start/end for lossless image_to_maze).
DEFAULT_WALL = (0, 0, 0)
DEFAULT_FREE = (255, 255, 255)
DEFAULT_START = (0, 255, 0)
DEFAULT_END = (0, 0, 255)
DEFAULT_PATH = (255, 0, 0)

_FILENAME_RE = re.compile(r"^maze_(\d+)_(\d+)_(\d+)_(.+)\.png$")


def random_frame_openings(
    maze_gen: MazeGenerator, rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Pick two distinct random cells on the outer frame that are wall cells and
    orthogonally adjacent to at least one path cell (so opening them connects
    to the maze). Default start/end from generate_maze are sealed first.
    """
    h, w = maze_gen.height, maze_gen.width
    maze = maze_gen.maze

    old_start, old_end = maze_gen.start_pos, maze_gen.end_pos
    maze[old_start[0]][old_start[1]] = 1
    maze[old_end[0]][old_end[1]] = 1

    candidates = []
    for r in range(h):
        for c in range(w):
            if not (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                continue
            if maze[r][c] != 1:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and maze[nr][nc] == 0:
                    candidates.append((r, c))
                    break

    if len(candidates) < 2:
        raise ValueError(
            "Not enough valid border openings; use a larger odd width/height."
        )

    a, b = rng.sample(candidates, 2)
    maze[a[0]][a[1]] = 0
    maze[b[0]][b[1]] = 0
    maze_gen.start_pos, maze_gen.end_pos = a, b
    return a, b


def _grid_edges(n: int, total: int) -> list[int]:
    """Partition ``total`` pixels into ``n`` columns/rows; edges are monotonic."""
    if n <= 0:
        return [0, total]
    edges = [round(i * total / n) for i in range(n + 1)]
    edges[0] = 0
    edges[-1] = total
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    return edges


def _cell_center_xy(
    r: int, c: int, x_edges: list[int], y_edges: list[int]
) -> tuple[float, float]:
    """Pixel center of the cell rectangle used by :meth:`Maze.maze_to_image`."""
    x0, x1 = x_edges[c], x_edges[c + 1] - 1
    y0, y1 = y_edges[r], y_edges[r + 1] - 1
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _nearest_label(rgb: tuple[int, int, int], palette: dict[str, tuple[int, int, int]]) -> str:
    best_k = ""
    best_d = float("inf")
    r, g, b = rgb
    for label, (pr, pg, pb) in palette.items():
        d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if d < best_d:
            best_d = d
            best_k = label
    return best_k


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _recover_path_order(arr: np.ndarray, s: tuple[int, int], e: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Greedy walk from S toward E using P cells (and E); if ambiguous or stuck,
    fall back to row-major order of P cells.
    """
    h, w = arr.shape
    p_cells = [(r, c) for r in range(h) for c in range(w) if arr[r, c] == "P"]
    if not p_cells:
        return []

    order: list[tuple[int, int]] = [s]
    visited: set[tuple[int, int]] = {s}
    cur = s

    while cur != e:
        nxt = None
        for nr, nc in _neighbors4(cur[0], cur[1], h, w):
            if (nr, nc) in visited:
                continue
            v = arr[nr, nc]
            if v == "E":
                order.append((nr, nc))
                return order
            if v == "P":
                if nxt is not None:
                    nxt = None
                    break
                nxt = (nr, nc)
        if nxt is None:
            break
        order.append(nxt)
        visited.add(nxt)
        cur = nxt

    if len(order) >= 2 and order[-1] == e:
        return order

    p_cells.sort()
    return p_cells


class Maze:
    """2D maze with wall/path/start/end/path markers and PNG round-trip."""

    default_wall: ClassVar[tuple[int, int, int]] = DEFAULT_WALL
    default_free: ClassVar[tuple[int, int, int]] = DEFAULT_FREE
    default_start: ClassVar[tuple[int, int, int]] = DEFAULT_START
    default_end: ClassVar[tuple[int, int, int]] = DEFAULT_END
    default_path: ClassVar[tuple[int, int, int]] = DEFAULT_PATH
    default_image_height: ClassVar[int] = 800

    def __init__(
        self, width: int, height: int, seed: int, image_height: int | None = None
    ) -> None:
        self.seed = seed
        self.image_height = (
            int(image_height) if image_height is not None else self.default_image_height
        )
        maze_gen = MazeGenerator(width=width, height=height, seed=seed)
        maze_gen.generate_maze()

        self.width = maze_gen.width
        self.height = maze_gen.height

        frame_rng = random.Random(maze_gen.seed if maze_gen.seed is not None else seed)
        start, end = random_frame_openings(maze_gen, frame_rng)

        h, w = self.height, self.width
        self.array = np.empty((h, w), dtype=object)
        for r in range(h):
            for c in range(w):
                v = maze_gen.maze[r][c]
                self.array[r, c] = 1 if v == 1 else 0

        self.array[start[0], start[1]] = "S"
        self.array[end[0], end[1]] = "E"
        self.path: list[tuple[int, int]] = []
        self.path_image_tag: str = "none"

    def find_final_and_exploratory_paths(
        self, use_bfs: bool = False
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Search from S to E with breadth-first (``use_bfs=True``) or depth-first
        (default).

        Returns ``(final_path, exploratory_path)``. With BFS, ``final_path`` is a
        shortest route; ``exploratory_path`` is dequeue order. With DFS,
        ``final_path`` is *a* route (neighbor order from :func:`_neighbors4`,
        reversed for stack order). ``exploratory_path`` is a **continuous** walk:
        each step is to an orthogonal neighbor, including backtracking along the
        DFS tree after a failed subtree.
        """
        h, w = self.height, self.width
        arr = self.array

        start = end = None
        for r in range(h):
            for c in range(w):
                v = arr[r, c]
                if v == "S":
                    start = (r, c)
                elif v == "E":
                    end = (r, c)
        if start is None or end is None:
            return [], []

        def walkable(r: int, c: int) -> bool:
            return arr[r, c] != 1

        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        exploratory_path: list[tuple[int, int]] = []

        if use_bfs:
            queue: deque[tuple[int, int]] = deque([start])
            while queue:
                cell = queue.popleft()
                exploratory_path.append(cell)
                if cell == end:
                    break
                r, c = cell
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) in parent:
                        continue
                    if not walkable(nr, nc):
                        continue
                    parent[nr, nc] = cell
                    queue.append((nr, nc))
        else:
            # Iterative DFS with explicit frames so exploratory_path records every
            # move (forward + backtrack along parent links), matching a continuous walk.
            stack_frames: list[
                tuple[tuple[int, int], int, list[tuple[int, int]]]
            ] = []
            neighbors_start = list(
                reversed(_neighbors4(start[0], start[1], h, w))
            )
            stack_frames.append((start, 0, neighbors_start))
            exploratory_path.append(start)

            while stack_frames:
                u, i, neighbors = stack_frames[-1]
                if i >= len(neighbors):
                    popped = stack_frames.pop()
                    if not stack_frames:
                        break
                    child = popped[0]
                    parent_u = stack_frames[-1][0]
                    walk = parent[child]
                    while walk is not None and walk != parent_u:
                        exploratory_path.append(walk)
                        walk = parent[walk]
                    exploratory_path.append(parent_u)
                    continue
                v = neighbors[i]
                stack_frames[-1] = (u, i + 1, neighbors)
                if v in parent or not walkable(v[0], v[1]):
                    continue
                parent[v] = u
                exploratory_path.append(v)
                if v == end:
                    break
                stack_frames.append(
                    (
                        v,
                        0,
                        list(reversed(_neighbors4(v[0], v[1], h, w))),
                    )
                )

        if end not in parent:
            return [], exploratory_path

        final_path: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = end
        while cur is not None:
            final_path.append(cur)
            cur = parent[cur]
        final_path.reverse()
        return final_path, exploratory_path

    def draw_solution_path(
        self,
        exploratory_path: bool = False,
        *,
        use_bfs: bool = False,
    ) -> None:
        """
        Run BFS or DFS (default) and mark path cells as ``P`` in :attr:`array`.

        By default only the found solution path is marked. With
        ``exploratory_path=True``, every cell in search visit order is marked
        (BFS dequeue order, or DFS as a continuous walk with backtracking). Start and end stay as
        ``S`` / ``E``. Clears previous ``P`` markers first. Updates :attr:`path`
        to the coordinates that were drawn and :attr:`path_image_tag` for
        :meth:`maze_to_image` filenames (e.g. ``DFS_final``, ``BFS_exploratory``).
        """
        final_path, visit_order = self.find_final_and_exploratory_paths(use_bfs=use_bfs)
        coords = visit_order if exploratory_path else final_path

        self.path_image_tag = (
            f"{'BFS' if use_bfs else 'DFS'}_"
            f"{'exploratory' if exploratory_path else 'final'}"
        )

        arr = self.array
        for r in range(self.height):
            for c in range(self.width):
                if arr[r, c] == "P":
                    arr[r, c] = 0

        for r, c in coords:
            v = arr[r, c]
            if v == "S" or v == "E":
                continue
            if v != 1:
                arr[r, c] = "P"

        self.path = list(coords)

    def maze_to_image(
        self,
        output_folder: str,
        *,
        wall: tuple[int, int, int] | None = None,
        free: tuple[int, int, int] | None = None,
        start: tuple[int, int, int] | None = None,
        end: tuple[int, int, int] | None = None,
        path: tuple[int, int, int] | None = None,
        path_line_width_fraction: float = 0.25,
    ) -> str:
        """
        Rasterize ``self.array`` to a PNG. Output height is ``self.image_height``
        pixels; each row uses ``image_height / height`` pixels on average. Width
        scales so cells stay square (same aspect as the maze grid).

        Path cells (``P``) are filled with the free color; the solution polyline from
        :attr:`path` is drawn in the path color with a dot at each cell center and
        segments joining consecutive centers. Stroke width is
        ``path_line_width_fraction`` times the pixel width of a column (default 1/4).

        Round-trip with ``image_to_maze`` is reliable when using the default palette
        (or when the same RGB tuples are used for decoding — defaults only for v1).
        """
        wall = wall if wall is not None else self.default_wall
        free = free if free is not None else self.default_free
        start_c = start if start is not None else self.default_start
        end_c = end if end is not None else self.default_end
        path_c = path if path is not None else self.default_path

        os.makedirs(output_folder, exist_ok=True)
        fname = f"maze_{self.width}_{self.height}_{self.seed}_{self.path_image_tag}.png"
        out_path = os.path.join(output_folder, fname)

        img_h = self.image_height
        img_w = int(round(img_h * self.width / self.height))
        x_edges = _grid_edges(self.width, img_w)
        y_edges = _grid_edges(self.height, img_h)
        im = Image.new("RGB", (img_w, img_h), free)
        draw = ImageDraw.Draw(im)

        arr = self.array
        for r in range(self.height):
            for c in range(self.width):
                v = arr[r, c]
                if v == 1:
                    color = wall
                elif v == 0:
                    color = free
                elif v == "S":
                    color = start_c
                elif v == "E":
                    color = end_c
                elif v == "P":
                    color = free
                else:
                    color = free
                x0, y0 = x_edges[c], y_edges[r]
                x1, y1 = x_edges[c + 1] - 1, y_edges[r + 1] - 1
                draw.rectangle([x0, y0, x1, y1], fill=color, outline=color)

        if self.path:
            col_w = x_edges[1] - x_edges[0] if self.width > 0 else 1
            line_w = max(1, int(round(path_line_width_fraction * col_w)))
            dot_r = max(1.0, line_w / 2.0)
            centers = [
                _cell_center_xy(r, c, x_edges, y_edges) for r, c in self.path
            ]
            for i in range(1, len(centers)):
                draw.line([centers[i - 1], centers[i]], fill=path_c, width=line_w)
            for cx, cy in centers:
                x0 = int(cx - dot_r)
                y0 = int(cy - dot_r)
                x1 = int(cx + dot_r)
                y1 = int(cy + dot_r)
                draw.ellipse([x0, y0, x1, y1], fill=path_c, outline=path_c)

        im.save(out_path, format="PNG")
        return out_path

    @classmethod
    def image_to_maze(cls, image_path: str) -> Maze:
        """
        Load a PNG written by :meth:`maze_to_image`. Parses ``width``, ``height``,
        and ``seed`` from the filename. Decodes pixels using the default palette
        (nearest RGB). ``self.path`` is filled by walking ``P`` cells from ``S`` to
        ``E`` when possible; otherwise row-major ``P`` order.
        """
        base = os.path.basename(image_path)
        m = _FILENAME_RE.match(base)
        if not m:
            raise ValueError(
                f"Filename must match maze_<width>_<height>_<seed>_<tag>.png; got {base!r}"
            )
        width, height, seed_s, tag = m.groups()
        w, h = int(width), int(height)
        seed = int(seed_s)

        im = Image.open(image_path).convert("RGB")
        img_w, img_h = im.size
        x_edges = _grid_edges(w, img_w)
        y_edges = _grid_edges(h, img_h)

        palette = {
            "wall": cls.default_wall,
            "free": cls.default_free,
            "start": cls.default_start,
            "end": cls.default_end,
            "path": cls.default_path,
        }

        obj = cls.__new__(cls)
        obj.width = w
        obj.height = h
        obj.seed = seed
        obj.image_height = img_h
        obj.array = np.empty((h, w), dtype=object)
        obj.path = []
        obj.path_image_tag = tag

        px = im.load()
        for r in range(h):
            for c in range(w):
                cx = (x_edges[c] + x_edges[c + 1] - 1) // 2
                cy = (y_edges[r] + y_edges[r + 1] - 1) // 2
                rgb = px[cx, cy]
                label = _nearest_label(rgb, palette)
                if label == "wall":
                    obj.array[r, c] = 1
                elif label == "free":
                    obj.array[r, c] = 0
                elif label == "start":
                    obj.array[r, c] = "S"
                elif label == "end":
                    obj.array[r, c] = "E"
                else:
                    obj.array[r, c] = "P"

        s_pos = e_pos = None
        for r in range(h):
            for c in range(w):
                if obj.array[r, c] == "S":
                    s_pos = (r, c)
                elif obj.array[r, c] == "E":
                    e_pos = (r, c)
        if s_pos is None or e_pos is None:
            obj.path = []
            return obj

        obj.path = _recover_path_order(obj.array, s_pos, e_pos)
        return obj
