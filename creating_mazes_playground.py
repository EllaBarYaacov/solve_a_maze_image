import os
from datetime import datetime

from maze import Maze

now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
ROOT = os.path.join(os.path.dirname(__file__), f"solution_seq_{now}")
os.makedirs(ROOT, exist_ok=True)

maze = Maze(9, 9, 42)
saved, csv_path = maze.create_solution_images_seq(ROOT, exploratory=False)

print(f"Wrote {len(saved)} frames to:\n  {ROOT}")
print(f"CSV: {csv_path}")
if saved:
    print("  first:", os.path.basename(saved[0]))
    print("  last: ", os.path.basename(saved[-1]))
