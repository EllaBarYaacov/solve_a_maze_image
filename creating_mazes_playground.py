import os
from datetime import datetime

from maze import Maze

now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
ROOT = os.path.join(os.path.dirname(__file__), f"roundtrip_playground_{now}")
os.makedirs(ROOT, exist_ok=True)

seed = 42
size = 9
maze = Maze(size, size, seed)
maze.draw_solution_path(exploratory_path=False)

paths: list[str] = []
paths.append(maze.maze_to_image(output_folder=ROOT, extra_info="_i1"))
for i in range(2, 6):
    maze = Maze.image_to_maze(paths[-1])
    paths.append(maze.maze_to_image(output_folder=ROOT, extra_info=f"_i{i}"))

print(f"Wrote {len(paths)} images to {ROOT}:")
for p in paths:
    print(" ", os.path.basename(p))
