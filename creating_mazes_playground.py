import os
from datetime import datetime

from maze import Maze

height = 9
width = height
seed = 42

MAZES_ROOT = f"mazes_{width}x{height}_{seed}"

height = 29
width = height
maze = Maze(width=width, height=height, seed=seed)
now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
MAZES_ROOT = f"mazes_{width}x{height}_{seed}_{now}"

out_path = maze.maze_to_image(output_folder=MAZES_ROOT)
# maze.draw_solution_path()
# maze.maze_to_image(output_folder=MAZES_ROOT)
maze.draw_solution_path(exploratory_path=True)
maze.maze_to_image(output_folder=MAZES_ROOT)
print(f"Path: {maze.path}")

