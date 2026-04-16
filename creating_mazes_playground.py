import os
from datetime import datetime
from maze import Maze


seed = 42
height = 9
width = height
maze = Maze(width=width, height=height, seed=seed)
now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
MAZES_ROOT = f"mazes_{width}x{height}_{seed}_{now}"

out_path = maze.maze_to_image(output_folder=MAZES_ROOT)

for use_bfs in [False, True]:
    for exploratory_path in [False, True]:
        maze.draw_solution_path(use_bfs=use_bfs, exploratory_path=exploratory_path)
        out_path = maze.maze_to_image(output_folder=MAZES_ROOT)
        print(f"Use BFS: {use_bfs}, Exploratory Path: {exploratory_path}")
        print(f"Path: {maze.path}")

