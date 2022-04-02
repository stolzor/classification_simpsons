from pathlib import Path
import numpy as np
from preproc_img import SimpsonsDataset
from sklearn.model_selection import train_test_split


"""
    Inside the folder "simpsons_dataset" contains the same folder "simpsons_dataset",
    which should be deleted because it duplicates the same data.
    data/simpsons_dataset/simpsons_dataset/
"""
# shutil.rmtree('data/simpsons_dataset/simpsons_dataset/') # delete double folder data
train_files_path = Path("data/simpsons_dataset/")  # data path
files = list(train_files_path.rglob('*.jpg'))
labels = np.unique([path.parent.name for path in files])

train_files_path, valid_files_path = train_test_split(files, train_size=0.8, stratify=[path.parent.name for path in files])

data_train = SimpsonsDataset(train_files_path, 'train')
data_valid = SimpsonsDataset(valid_files_path, 'valid')