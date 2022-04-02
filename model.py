from torchvision import models
import torch

model = models.regnet_x_800mf()
for i, param in enumerate(model.parameters()):
    if i <= 90:
        param.requires_grad = False
num_class = 42
model.fc = torch.nn.Linear(672, num_class)