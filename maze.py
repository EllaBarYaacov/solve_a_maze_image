"""Maze grid with labeled cells, PNG export, and round-trip image loading."""

from __future__ import annotations

import csv
import json
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
        self,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Depth-first search from S to E.
        Returns ``(final_path, exploratory_path)``. ``final_path`` is *a* route
        ``exploratory_path`` is a **continuous** walk: each step is to an
        orthogonal neighbor, including backtracking along the DFS tree after a
        failed subtree.
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
    ) -> None:
        """
        Run DFS and mark path cells as ``P`` in :attr:`array`.

        By default only the found solution path is marked. 
        With ``exploratory_path=True``, every cell in search visit order is marked
        (DFS as a continuous walk with backtracking). Start and end stay as
        ``S`` / ``E``. Clears previous ``P`` markers first. Updates :attr:`path`
        to the coordinates that were drawn and :attr:`path_image_tag` for
        :meth:`maze_to_image` filenames (e.g. ``DFS_final``, ``DFS_exploratory``).
        """
        final_path, visit_order = self.find_final_and_exploratory_paths()
        coords = visit_order if exploratory_path else final_path

        self.path_image_tag = (
            f"DFS_{'exploratory' if exploratory_path else 'final'}"
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

    def create_solution_images_seq(
        self,
        output_folder: str,
        *,
        exploratory: bool = False,
        path_line_width_fraction: float = 0.25,
    ) -> tuple[list[str], str]:
        """
        Save a sequence of PNGs under ``output_folder``. Frame ``k`` (suffix ``_step{k:04d}``)
        shows the walk **through the first ``k`` positions**; the last frame shows the
        full chosen sequence.

        If ``exploratory`` is false, the sequence is the **final** S→E solution path and
        frames use the same drawing as ``DFS_final`` in :meth:`maze_to_image`. If true,
        the sequence follows the **DFS exploratory** visit order (including backtracking)
        and uses ``DFS_exploratory`` styling (quarter-cell anchors). A one-cell prefix
        falls back to ``DFS_final`` rendering so stroke geometry stays valid.

        Also writes a CSV beside the PNGs with columns ``image_path``, ``solution_pixels``
        (JSON list of ``[x, y]`` polyline vertices in image pixels, matching the red
        stroke), and ``line_width`` (pixel stroke width). Returns ``(list of image paths,
        path to the csv file)``.
        """
        final_path, exploratory_path = self.find_final_and_exploratory_paths()
        seq = exploratory_path if exploratory else final_path
        if not seq:
            return [], ""

        arr = self.array
        out_paths: list[str] = []
        csv_rows: list[tuple[str, str, int]] = []

        for step in range(1, len(seq) + 1):
            prefix = seq[:step]
            for r in range(self.height):
                for c in range(self.width):
                    if arr[r, c] == "P":
                        arr[r, c] = 0
            for r, c in prefix:
                v = arr[r, c]
                if v not in ("S", "E") and v != 1:
                    arr[r, c] = "P"

            self.path = prefix
            if exploratory and len(prefix) >= 2:
                self.path_image_tag = "DFS_exploratory"
            else:
                self.path_image_tag = "DFS_final"

            out_path = self.maze_to_image(
                output_folder,
                extra_info=f"_step{step:04d}",
                path_line_width_fraction=path_line_width_fraction,
            )
            out_paths.append(out_path)

            verts, lw = _solution_polyline_pixels_and_line_width(
                prefix,
                self.path_image_tag,
                self.width,
                self.height,
                self.image_height,
                path_line_width_fraction,
            )
            pix_json = json.dumps(
                [[round(x, 3), round(y, 3)] for x, y in verts],
                separators=(",", ":"),
            )
            csv_rows.append((os.path.abspath(out_path), pix_json, lw))

        csv_fname = (
            f"maze_{self.width}_{self.height}_{self.seed}_"
            f"{'exploratory' if exploratory else 'final'}_solution_seq.csv"
        )
        csv_path = os.path.join(output_folder, csv_fname)
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["image_path", "solution_pixels", "line_width"])
            writer.writerows(csv_rows)

        return out_paths, csv_path

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
        extra_info: str = "",
    ) -> str:
        """
        Rasterize ``self.array`` to a PNG. Output height is ``self.image_height``
        pixels; each row uses ``image_height / height`` pixels on average. Width
        scales so cells stay square (same aspect as the maze grid).

        Path cells (``P``) are filled with the free color. For tags containing
        ``exploratory``, the polyline uses quarter-cell centers from
        :func:`_choose_point_quadrant`, drawn segment by segment. For the final path,
        a single polyline through **cell centers** uses ``joint="curve"``. Stroke width
        is ``path_line_width_fraction`` times the pixel width of a column (default 1/4).

        The path stroke is drawn **on top** of the grid (including through start/end
        cells), matching the original look: red connects visibly through green and blue.
        :meth:`image_to_maze` uses multi-sample decoding so ``S``/``E`` are still read
        correctly (corners of those cells keep the marker color beside the stroke).

        ``extra_info`` is appended to the filename before ``.png`` (e.g. ``_i2``), default
        empty so existing names stay ``maze_<w>_<h>_<seed>_<tag>.png``.

        Round-trip with ``image_to_maze`` is reliable when using the default palette
        (or when the same RGB tuples are used for decoding — defaults only for v1).
        """
        wall = wall if wall is not None else self.default_wall
        free = free if free is not None else self.default_free
        start_c = start if start is not None else self.default_start
        end_c = end if end is not None else self.default_end
        path_c = path if path is not None else self.default_path

        os.makedirs(output_folder, exist_ok=True)
        fname = (
            f"maze_{self.width}_{self.height}_{self.seed}_"
            f"{self.path_image_tag}{extra_info}.png"
        )
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
            vertices, line_w = _solution_polyline_pixels_and_line_width(
                self.path,
                self.path_image_tag,
                self.width,
                self.height,
                self.image_height,
                path_line_width_fraction,
            )
            if len(vertices) >= 2:
                draw.line(vertices, fill=path_c, width=line_w, joint="curve")

        im.save(out_path, format="PNG")
        return out_path

    @classmethod
    def image_to_maze(cls, image_path: str) -> Maze:
        """
        This method is used to restore a maze from an image.
        IT WORKS ONLY FOR FINAL PATHS (not exploratory paths).
        Load a PNG written by :meth:`maze_to_image`. Parses ``width``, ``height``,
        and ``seed`` from the filename. Each cell is decoded with several RGB samples
        per cell (see :func:`_decode_grid_cell_from_pixels`) so start/end/path stay
        distinct even when the red stroke covers the cell center. ``self.path`` is
        filled by walking ``P`` cells from ``S`` to ``E`` when possible; otherwise
        row-major ``P`` order.
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
                obj.array[r, c] = _decode_grid_cell_from_pixels(
                    px, r, c, x_edges, y_edges, palette
                )

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



def random_frame_openings(
    maze_gen: MazeGenerator, rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Pick two distinct random cells on the outer frame that are wall cells and
    orthogonally adjacent to at least one path cell (each opening connects
    to the maze). Default start/end from generate_maze are sealed first.

    Start and end are placed on **opposite** edges: either top vs bottom or
    left vs right (chosen at random when both are possible).
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

    on_top = [p for p in candidates if p[0] == 0]
    on_bottom = [p for p in candidates if p[0] == h - 1]
    on_left = [p for p in candidates if p[1] == 0]
    on_right = [p for p in candidates if p[1] == w - 1]

    orientations: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    if h > 1 and on_top and on_bottom:
        orientations.append((on_top, on_bottom))
    if w > 1 and on_left and on_right:
        orientations.append((on_left, on_right))

    if orientations:
        pool_a, pool_b = rng.choice(orientations)
        a = rng.choice(pool_a)
        b = rng.choice(pool_b)
        if a == b:
            a, b = rng.sample(candidates, 2)
    else:
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
    """Pixel center of grid cell ``(row, column)`` for the final solution path."""
    x0 = float(x_edges[c])
    x1 = float(x_edges[c + 1] - 1)
    y0 = float(y_edges[r])
    y1 = float(y_edges[r + 1] - 1)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _cell_four_quadrant_centers_xy(
    r: int, c: int, x_edges: list[int], y_edges: list[int]
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """
    Centers of the four sub-squares when the cell is split into a 2×2 grid (``y``
    increases downward).

    Returns ``(up_left, up_right, down_left, down_right)`` — center of the top-left,
    top-right, bottom-left, and bottom-right quarter of the cell rectangle used by
    :meth:`Maze.maze_to_image`.
    """
    x0 = float(x_edges[c])
    x1 = float(x_edges[c + 1] - 1)
    y0 = float(y_edges[r])
    y1 = float(y_edges[r + 1] - 1)
    mx = (x0 + x1) / 2.0
    my = (y0 + y1) / 2.0
    up_left = ((x0 + mx) / 2.0, (y0 + my) / 2.0)
    up_right = ((mx + x1) / 2.0, (y0 + my) / 2.0)
    down_left = ((x0 + mx) / 2.0, (my + y1) / 2.0)
    down_right = ((mx + x1) / 2.0, (my + y1) / 2.0)
    return up_left, up_right, down_left, down_right

def _choose_point_quadrant(path: list[tuple[int, int]], index: int, up_left, up_right, down_left, down_right) -> int:
    if index == 0:
        first_cell = path[index]
        second_cell = path[index + 1]
        third_cell = None
    elif index == len(path) - 1:
        first_cell = path[index - 1]
        second_cell = path[index]
        third_cell = None
    else:
        first_cell = path[index-1]
        second_cell = path[index]
        third_cell = path[index + 1]

    first_2_cells_direction = _get_2_cells_direction(first_cell, second_cell)

    if third_cell is None:
        if first_2_cells_direction == "up":
            return [up_right], ["UR"]
        elif first_2_cells_direction == "down":
            return [down_left], ["DL"]
        elif first_2_cells_direction == "left":
            return [up_left], ["UL"]
        elif first_2_cells_direction == "right":
            return [down_right], ["DR"]



    last_3_cells_maneuver = _get_last_3_cells_maneuver(first_cell, second_cell, third_cell)

    if first_2_cells_direction == "up":
        if last_3_cells_maneuver == "forward" or last_3_cells_maneuver == "left turn":
            return [up_right], ["UR"]
        elif last_3_cells_maneuver == "right turn":
            return [down_right], ["DR"]
        elif last_3_cells_maneuver == "u turn":
            return [up_right, up_left], ["UR", "UL"]
    elif first_2_cells_direction == "down":
        if last_3_cells_maneuver == "forward" or last_3_cells_maneuver == "left turn":
            return [down_left], ["DL"]
        elif last_3_cells_maneuver == "right turn":
            return [up_left], ["UL"]
        elif last_3_cells_maneuver == "u turn":
            return [down_left, down_right], ["DL", "DR"]
    elif first_2_cells_direction == "left":
        if last_3_cells_maneuver == "forward" or last_3_cells_maneuver == "left turn":
            return [up_left], ["UL"]
        elif last_3_cells_maneuver == "right turn":
            return [up_right], ["UR"]
        elif last_3_cells_maneuver == "u turn":
            return [up_left, down_left], ["UL", "DL"]
    elif first_2_cells_direction == "right":
        if last_3_cells_maneuver == "forward" or last_3_cells_maneuver == "left turn":
            return [down_right], ["DR"]
        elif last_3_cells_maneuver == "right turn":
            return [down_left], ["DL"]
        elif last_3_cells_maneuver == "u turn":
            return [down_right, up_right], ["DR", "UR"]


def _solution_polyline_pixels_and_line_width(
    path: list[tuple[int, int]],
    path_image_tag: str,
    grid_width: int,
    grid_height: int,
    image_height: int,
    path_line_width_fraction: float,
) -> tuple[list[tuple[float, float]], int]:
    """
    Same path geometry as :meth:`Maze.maze_to_image` (vertices and stroke width in pixels).
    """
    img_h = image_height
    img_w = int(round(img_h * grid_width / grid_height))
    x_edges = _grid_edges(grid_width, img_w)
    y_edges = _grid_edges(grid_height, img_h)
    col_w = x_edges[1] - x_edges[0] if grid_width > 0 else 1
    line_w = max(1, int(round(path_line_width_fraction * col_w)))
    vertices: list[tuple[float, float]] = []
    if not path:
        return [], line_w
    if "exploratory" in path_image_tag:
        for i, (r, c) in enumerate(path):
            up_left, up_right, down_left, down_right = _cell_four_quadrant_centers_xy(
                r, c, x_edges, y_edges
            )
            chosen, _ = _choose_point_quadrant(
                path, i, up_left, up_right, down_left, down_right
            )
            vertices.extend(chosen)
    elif len(path) >= 2:
        vertices = [
            _cell_center_xy(r, c, x_edges, y_edges) for r, c in path
        ]
    return vertices, line_w


def _get_last_3_cells_maneuver(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> str:
    """
    a, b, c: (row, column) consecutive cells on a path (same order as
    :func:`_get_2_cells_direction`). Returns how you turn at b when going a -> b -> c.
    """
    first_direction = _get_2_cells_direction(a, b)
    second_direction = _get_2_cells_direction(b, c)
    if first_direction == second_direction:
        return "forward"
    left_turn_directions = [("up", "left"), ("down", "right"), ("left", "down"), ("right", "up")]
    right_turn_directions = [("up", "right"), ("down", "left"), ("left", "up"), ("right", "down")]
    if (first_direction, second_direction) in left_turn_directions:
        return "left turn"
    elif (first_direction, second_direction) in right_turn_directions:
        return "right turn"
    else:
        return "u turn"


def _get_2_cells_direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    d_row, d_col = b[0] - a[0], b[1] - a[1]
    if d_col == 1:  return "right"
    if d_col == -1: return "left"
    if d_row == 1:  return "down"   
    if d_row == -1: return "up"
    raise ValueError("not a single orthogonal step")

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


def _decode_grid_cell_from_pixels(
    px,
    r: int,
    c: int,
    x_edges: list[int],
    y_edges: list[int],
    palette: dict[str, tuple[int, int, int]],
) -> object:
    """
    Map one maze cell to ``1``, ``0``, ``"S"``, ``"E"``, or ``"P"`` using several
    RGB samples (3×3 lattice in pixel space).

    A single center sample fails when the red path stroke sits on top of start/end
    (center reads as ``path``). Sampling corners can still see green/blue. Priority
    ``start`` / ``end`` over ``path`` fixes that for the usual stroke width.
    """
    x0 = x_edges[c]
    x1 = x_edges[c + 1] - 1
    y0 = y_edges[r]
    y1 = y_edges[r + 1] - 1
    xs = sorted({x0, (x0 + x1) // 2, x1})
    ys = sorted({y0, (y0 + y1) // 2, y1})
    labels: list[str] = []
    for py in ys:
        for px_i in xs:
            labels.append(_nearest_label(px[px_i, py], palette))
    if "start" in labels:
        return "S"
    if "end" in labels:
        return "E"
    if "path" in labels:
        return "P"
    if labels.count("wall") > labels.count("free"):
        return 1
    return 0


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

