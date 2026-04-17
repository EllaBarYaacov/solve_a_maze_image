import os
from datetime import datetime
from maze import Maze


seed = 42
height = 7
width = height
maze = Maze(width=width, height=height, seed=seed)
now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
MAZES_ROOT = f"mazes_{width}x{height}_{seed}_{now}"

out_path = maze.maze_to_image(output_folder=MAZES_ROOT)
maze.draw_solution_path(exploratory_path=True)
out_path = maze.maze_to_image(output_folder=MAZES_ROOT)
maze.draw_solution_path(exploratory_path=False)
out_path = maze.maze_to_image(output_folder=MAZES_ROOT)

