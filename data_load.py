from pathlib import Path
import numpy as np

train_files_path = Path("data/train/")  # data path
files = list(train_files_path.rglob('*.jpg'))
labels = np.unique([path.parent.name for path in files])
print(len(files), len(labels))