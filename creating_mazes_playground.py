import os
from datetime import datetime
from maze import Maze


now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
ROOT = f"restored_maze_{now}"
image_path = "20260417_134734_206505_mazes_9x9_42/maze_9_9_42_DFS_final.png"
restored_maze = Maze.image_to_maze(image_path)
print(f"restored_maze.array: \n{restored_maze.array}")
out_path = restored_maze.maze_to_image(output_folder=ROOT)