# Created by MacBook Pro at 14.07.25

from pathlib import Path

# config.py

# Size variants for shape
size_list = ["s", "m", "l"]
shapes = ['circle', 'square', 'triangle']
colors = ['blue', 'green', 'orange']

# Whether to include principle in negative samples
prin_in_neg = False

root = Path(__file__).parents[0]
raw_patterns = root / 'video_tasks'

output_dir = root / "output"
if not output_dir.exists():
    output_dir.mkdir(parents=True)