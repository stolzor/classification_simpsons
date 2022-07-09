from torchvision import models
import torch

model = models.regnet_x_800mf()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_class = 42
model.fc = torch.nn.Linear(672, num_class)

if __name__ != 'model.model':
    model.load_state_dict(torch.load("weight_cnn/before_regnet_x_800mf.pt", map_location=device))
    for i, param in enumerate(model.parameters()):
        if i <= 90:
            param.requires_grad = False


