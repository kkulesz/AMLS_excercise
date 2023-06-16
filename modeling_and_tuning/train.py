from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch

from data_preparation.dataset import SdssDataset
from models.unet import UNet
from manual_validate import validate_model
import utils
import const

device = utils.get_device()


def train_single_epoch(model, loader, optimizer, criterion):
    for i, (input_tensor, target_tensor) in enumerate(loader):
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        prediction_tensor = model(input_tensor)

        loss = criterion(prediction_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    mo = UNet(in_channels=const.INPUT_CHANNELS, out_channels=const.OUTPUT_CHANNELS, bilinear=const.BILINEAR).to(device)

    dataset = SdssDataset(const.PIECES_READY_DATA_INPUTS_DIR, const.PIECES_READY_DATA_TARGETS_DIR)
    dataloader = DataLoader(dataset, batch_size=const.BATCH_SIZE, shuffle=True)
    opt = Adam(mo.parameters(), lr=const.LEARNING_RATE, betas=const.ADAM_BETAS)
    crt = nn.CrossEntropyLoss() if const.OUTPUT_CHANNELS > 1 else nn.BCEWithLogitsLoss()

    for epoch in range(const.NUMBER_OF_EPOCHS):
        print(f"epoch={epoch}")
        train_single_epoch(mo, dataloader, opt, crt)
    torch.save(mo.state_dict(), "model.pt")
    validate_model(model=mo)
