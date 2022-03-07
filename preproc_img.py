from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pickle


# SimpsonsDataset wrapper for data load
class SimpsonsDataset(Dataset):
    def __init__(self, files):
        super().__init__()
        self.files = sorted(files)

        self.label_encoder = LabelEncoder()

        self.labels = [path.parent.name for path in self.files]
        self.label_encoder.fit(self.labels)

        with open('label_encoder.pkl', 'wb') as lb_file:
            pickle.dump(self.label_encoder, lb_file)

    def __len__(self):
        return len(self.files)

    def load_img(self, img):
        image = Image.open(img)
        image.load()
        return image

    def prepare_sample(self, img):
        image = img.resize((244, 244))
        return np.array(image)

    def __getitem__(self, item):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = load_img(self.files[item])
        x = self.prepare_sample(x)
        x = transform(x)
        y = self.label_encoder.transform([self.labels[item]])
        y = y.item()
        return x, y


# evaluation functions
def load_img(img):
    image = Image.open(img)
    image.load()
    return image


def prepare_sample(img):
    image = img.resize((244, 244))
    return np.array(image)


def pre_process_img(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = load_img(img)
    x = prepare_sample(x)
    x = transform(x)
    return x