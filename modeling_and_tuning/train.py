from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np

from data_preparation.dataset import SdssDataset
from models.unet import UNet
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
    mo = UNet(in_channels=const.INPUT_CHANNELS, out_channels=const.OUTPUT_CHANNELS, bilinear=True).to(device)

    dataset = SdssDataset(const.PIECES_READY_DATA_INPUTS_DIR, const.PIECES_READY_DATA_TARGETS_DIR)
    dataloader = DataLoader(dataset, batch_size=const.BATCH_SIZE, shuffle=True)
    opt = Adam(mo.parameters(), lr=const.LEARNING_RATE, betas=const.ADAM_BETAS)
    crt = nn.CrossEntropyLoss() if const.OUTPUT_CHANNELS > 1 else nn.BCEWithLogitsLoss()

    for epoch in range(const.NUMBER_OF_EPOCHS):
        print(epoch)
        train_single_epoch(mo, dataloader, opt, crt)

    with torch.no_grad():
        inp, tgt = tuple(next(iter(dataloader)))
        inp_to_display = inp.numpy()[0]
        tgt = tgt.numpy()[0]
        iCh, iH, iW = inp_to_display.shape
        tCh, tH, tW = tgt.shape

        inp_to_display = np.reshape(inp_to_display, (iH, iW, iCh))
        tgt = np.reshape(tgt, (tH, tW, tCh))

        result = mo(inp.cuda())
        result = result.cpu().numpy()[0]
        result = np.reshape(result, (tH, tW, tCh))

        utils.display_image(inp_to_display)
        utils.display_image(tgt)
        utils.display_image(result)
