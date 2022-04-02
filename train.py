import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from data import data_train, data_valid
from model import model


def fit_epoch(model, criterion, optimizer, data, device):
    running_loss = 0
    running_acc = 0
    div = 0

    for inputs, label in data:
        inputs = inputs.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()
        pred = torch.argmax(outputs, -1)
        running_loss += loss.item() * inputs.shape[0]
        running_acc += torch.sum(pred == label.data)
        div += inputs.size(0)

    running_loss = running_loss / div
    running_acc = running_acc / div

    return running_loss, running_acc


def eval_epoch(model, criterion, data, device):
    model.eval()
    running_loss = 0
    running_acc = 0
    div = 0

    for inputs, label in data:
        inputs = inputs.to(device)
        label = label.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, label)
            pred = torch.argmax(outputs, -1)

        running_loss += loss.item() * inputs.shape[0]
        running_acc += torch.sum(pred == label.data)
        div += inputs.size(0)

    running_loss = running_loss / div
    running_acc = running_acc / div

    model.train()
    return running_loss, running_acc


def train(model, criterion, scheduler, optimizer, train, valid, device, epoch, batch_size=32):
    data_train = DataLoader(train, batch_size=batch_size, shuffle=True)
    data_valid = DataLoader(valid, batch_size=batch_size, shuffle=False)

    best_acc = 0
    best_w = 0

    text_info = "\ntrain loss: {train_loss: 0.4f} train acc: {train_acc: 0.4f}\
         valid loss: {valid_loss: 0.4f} valid acc: {valid_acc: 0.4f}"
    with tqdm(desc="epoch", total=epoch) as epoch_upd:
        for epochs in range(epoch):
            train_loss, train_acc = fit_epoch(model, criterion, optimizer, data_train, device)
            valid_loss, valid_acc = eval_epoch(model, criterion, data_valid, device)

            epoch_upd.update(1)
            tqdm.write(text_info.format(train_loss=train_loss, train_acc=train_acc, valid_loss=valid_loss,
                                        valid_acc=valid_acc))
            scheduler.step(valid_loss)

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_w = model.state_dict()

        torch.save(best_w, 'weight_cnn/after_regnet_x_800mf.pt')
        model.load_state_dict(best_w)


def pre_launch_setup():
    model.load_state_dict(torch.load("weight_cnn/before_regnet_x_800mf.pt"))

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.001,
                                                           patience=5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train(model, loss, scheduler, optimizer, data_train, data_valid, device, 2)


if __name__ == "__main__":
    print('-' * 7 + 'START TRAIN!' + '-' * 7)
    pre_launch_setup()

