from torchvision import models
import torch

model = models.regnet_x_800mf()
for i, param in enumerate(model.parameters()):
    if i <= 90:
        param.requires_grad = False
model.fc = torch.nn.Linear(672, 42)