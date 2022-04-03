import torch
from model import model
import numpy as np
import pickle
from data import pre_process_img
from pathlib import Path

def predict_one_sample(model, inputs, device):
    with torch.no_grad():
        model.eval()
        inputs = inputs.to(device)
        outputs = model(inputs).cpu()
        pred = torch.nn.functional.softmax(outputs, -1).numpy()
    return np.argmax(pred, -1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load("weight_cnn/after_regnet_x_800mf.pt", map_location=device))

    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    path = input()
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    inputs = pre_process_img(path, 'test')
    res = label_encoder.inverse_transform(predict_one_sample(model, inputs.unsqueeze(0), device))[0]
    print(' '.join([i.title() for i in res.split('_')]))