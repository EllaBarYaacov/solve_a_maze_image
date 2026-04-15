import os
from datetime import datetime

from maze import Maze

height = 9
width = height
seed = 42

MAZES_ROOT = f"mazes_{width}x{height}_{seed}"

for height in [7, 11, 15, 19, 23, 27, 31]:
    width = height
    maze = Maze(width=width, height=height, seed=seed)
    MAZES_ROOT = f"mazes_{width}x{height}_{seed}"
    for i in range(5):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = os.path.join(MAZES_ROOT, stamp)
        out_path = maze.maze_to_image(output_folder=out_dir)
