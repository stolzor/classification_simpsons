from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter
import numpy as np
import pickle
import torch

# SimpsonsDataset wrapper for data load
class SimpsonsDataset(Dataset):
    def __init__(self, files, mode, aug_len=None):
        super().__init__()
        self.labels = [path.parent.name for path in files]

        self.mode = mode

        self.files = files

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

        if self.mode == 'train':
            with open('label_encoder.pkl', 'wb') as lb_file:
                pickle.dump(self.label_encoder, lb_file)

    def __len__(self):
        return len(self.files)

    def load_img(self, img):
        image = Image.open(img)
        return self.prepare_sample(image)

    def prepare_sample(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((290, 290)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(img)
        return image.numpy().astype(np.float32)

    def transform_sample(self, img):
        """
        param img: image for transform
        return: transformed image
        """

        transform = {
            'Crop': transforms.Compose([
                transforms.Resize((310, 310)),
                transforms.CenterCrop((305, 305)),
                transforms.RandomCrop((290, 290))
            ]),
            'Rotate': transforms.Compose([
                transforms.RandomRotation((-25, 25))
            ]),
            'Hflip': transforms.Compose([
                transforms.RandomHorizontalFlip(p=1)
            ])
        }
        transform_list = list(transform.keys())

        augmenter = transform[transform_list[np.random.randint(3)]]
        aug_img = augmenter(img)

        return aug_img

    def __getitem__(self, item):
        img = torch.from_numpy(self.load_img(self.files[item]))
        x = self.transform_sample(img) if self.mode == 'train' else img

        if self.mode == 'test':
            return x

        y = self.label_encoder.transform([self.labels[item]]).item()
        return x, y


"""
    Inside the folder "simpsons_dataset" contains the same folder "simpsons_dataset",
    which should be deleted because it duplicates the same data.
    data/simpsons_dataset/simpsons_dataset/
"""
# shutil.rmtree('../data/simpsons_dataset/simpsons_dataset/') # delete double folder data
train_files_path = Path("../data/simpsons_dataset/")  # data path

files = list(train_files_path.rglob('*.jpg'))
labels = np.unique([path.parent.name for path in files])

def augmentation(img):
    transform = {
        'Crop': transforms.Compose([
            transforms.Resize((310, 310)),
            transforms.CenterCrop((305, 305)),
            transforms.RandomCrop((290, 290))
        ]),
        'Rotate': transforms.Compose([
            transforms.RandomRotation((-25, 25))
        ]),
        'Hflip': transforms.Compose([
            transforms.RandomHorizontalFlip(p=1)
        ])
    }
    transform_list = list(transform.keys())

    augmenter = transform[transform_list[np.random.randint(3)]]
    aug_img = augmenter(img)

    return aug_img

if False:
    num_aug = 1500
    to_add = Counter([path.parent for path in files])

    for name, value in to_add.items():
        add = (1500 - value)
        idx = 0
        last_num = 0
        while add != 0 and add > -1:
            add_zero = '0' * (4 - len(str(idx)))
            pattern_src = '{}\pic_{}{}.jpg'
            src = pattern_src.format(name, add_zero, idx)
            if  Path(src).is_file():
                add_zero = '0' * (4 - len(str(last_num)))
                img = Image.open(src)
                img.load()
                img = augmentation(img)
                print(img, name, add, pattern_src.format(name, 'aug_'+add_zero, last_num))
                img.save(pattern_src.format(name, 'aug_'+add_zero, last_num))
                last_num += 1
                add -= 1
            idx = 0 if idx == value else idx + 1
else:
    train_files_path = Path("../data/simpsons_dataset/")  # data path

    files = list(train_files_path.rglob('*.jpg'))
    labels = np.unique([path.parent.name for path in files])
    print(files)
    train_files_path, valid_files_path = train_test_split(files, train_size=0.8,
                                                      stratify=[path.parent.name for path in files])

    data_train = SimpsonsDataset(train_files_path, 'train')
    data_valid = SimpsonsDataset(valid_files_path, 'valid')