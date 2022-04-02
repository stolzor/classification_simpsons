from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pickle


# SimpsonsDataset wrapper for data load
class SimpsonsDataset(Dataset):
    def __init__(self, files, mode):
        super().__init__()
        self.labels = [path.parent.name for path in files]
        self.files = [self.load_img(img) for img in files]
        self.len_files = len(self.files)

        self.mode = mode

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

        if self.mode != 'test':
            with open('label_encoder.pkl', 'wb') as lb_file:
                pickle.dump(self.label_encoder, lb_file)

    def __len__(self):
        return 60000 if self.mode != 'test' and self.mode != 'val' else len(self.files)

    def load_img(self, img):
        image = Image.open(img)
        image.load()
        return image

    def prepare_sample(self, img):
        image = img.resize((244, 244))
        return np.array(image)

    def augmentation(self, img):
        """
        param img: image for augmentation
        return: augmented image
        """

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
        transform_list = list(transform.keys())

        augmenter = transform[transform_list[np.random.randint(3)]]

        aug_img = augmenter(img)

        return aug_img

    def __getitem__(self, item):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if self.mode != 'val' and self.mode != 'test':
            random_img = item if item < self.len_files else np.random.randint(1, self.len_files)
        else:
            random_img = item
        img = self.files[random_img]
        x = self.prepare_sample(self.augmentation(img)) if self.mode != 'test' and self.mode != 'val' else self.prepare_sample(img)
        x = transform(x)
        y = self.label_encoder.transform([self.labels[random_img]]).item()
        if self.mode == 'test':
            return x
        return x, y


# evaluation functions
def load_img(img):
    image = Image.open(img)
    image.load()
    return image


def prepare_sample(img):
    image = img.resize((244, 244))
    return np.array(image)


def pre_process_img(img, mode):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = load_img(img) if mode != 'train' else img
    x = prepare_sample(x) if mode != 'train' else img
    x = transform(x)
    return x


# convert Tensor to img
def from_array(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = (std * inp + mean) * 255
    return inp.astype(np.uint8)

