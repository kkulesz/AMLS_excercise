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
    in_channels = 5
    out_channels = 3
    mo = UNet(in_channels=in_channels, out_channels=out_channels).to(device)

    dataset = SdssDataset(const.PIECES_READY_DATA_INPUTS_DIR, const.PIECES_READY_DATA_TARGETS_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    opt = Adam(mo.parameters(), lr=0.02, betas=(0.5, 0.999))
    crt = nn.CrossEntropyLoss() if out_channels > 1 else nn.BCEWithLogitsLoss()

    # for epoch in range(2):
    #     train_single_epoch(mo, dataloader, opt, crt)

    # inp, tgt = tuple(next(iter(dataloader)))
    # inp = inp.detach().numpy().squeeze()
    # tgt = tgt.detach().numpy().squeeze()
    # iCh, iH, iW = inp.shape
    # tCh, tH, tW = tgt.shape
    #
    # inp = np.reshape(inp, (iH, iW, iCh))
    # tgt = np.reshape(tgt, (tH, tW, tCh))
    #
    # utils.display_image(inp)
    # utils.display_image(tgt)
    #
    # inp = torch.from_numpy(np.reshape(inp, (iCh, iH, iW)))
    # inp = torch.unsqueeze(inp, 0).to(device="cuda")
    # result = mo(inp)
    # print(result.shape)
    # result = result.detach().cpu().numpy().squeeze()
    # result = np.reshape(result, (tCh, tH, tW))
    # utils.display_image(result)
