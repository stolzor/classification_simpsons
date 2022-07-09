from PIL import Image
from torchvision.transforms import transforms
import numpy as np

# convert Tensor to img
def from_array(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = (std * inp + mean) * 255
    return inp.astype(np.uint8)

def load_img(img):
    image = Image.open(img).convert('RGB')
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