import os
from datetime import datetime

from maze import Maze

now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
ROOT = os.path.join(os.path.dirname(__file__), f"solution_seq_{now}")
os.makedirs(ROOT, exist_ok=True)

maze = Maze(9, 9, 42)
# Growing solution: _step0001.png … _stepNNNN.png (full path on last frame)
saved = maze.create_solution_images_seq(ROOT, exploratory=False)

