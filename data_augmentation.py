from torchvision import transforms
from data_load import labels, train_files_path
from preproc_img import SimpsonsDataset, load_img
import tqdm
import json
import numpy as np
from sklearn.model_selection import train_test_split

to_add = {}

with open('data/dict_path.json', 'r') as dp:
    dict_path = json.load(dp)

    for name in dict_path.keys():
        to_add[name] = 1700 - len(dict_path[name])
transform = {
    'Crop': transforms.Compose([
            transforms.Resize((310, 310)),
            transforms.CenterCrop((305, 305)),
            transforms.RandomCrop((270, 270))
    ]),
    'Rotate': transforms.Compose([
            transforms.RandomRotation((-25, 25))
    ]),
    'Hflip': transforms.Compose([
            transforms.RandomHorizontalFlip(p=1)
    ])
}
transform_list = ['Crop', 'Rotate', 'Hflip']

with tqdm.tqdm(total=42) as upd_b:
    for name in labels:
        for iter in range(to_add[name]):
            p_random = dict_path[name][np.random.randint(0, len(dict_path[name]))]

            image = load_img(p_random)
            label = p_random.split('\\')[-2]
            augmenter = transform[list(transform.keys())[np.random.randint(3)]]
            aug_img = augmenter(image)

            aug_img.save(f"{str(p_random)[:-13]}/aug_pic_{iter}.jpg")

        upd_b.update(1)

files = sorted(list(train_files_path.rglob(('*.jpg'))))
all_labels = sorted(list([path.parent.name for path in files]))

train, valid = train_test_split(files, train_size=0.85, stratify=all_labels)

data_train = SimpsonsDataset(train)
data_valid = SimpsonsDataset(valid)
