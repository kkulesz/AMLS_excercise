from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch

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
    # test = torch.randn(1, 5, 2048, 1489).to("cuda")
    # mo(test)

    dataset = SdssDataset(const.ALIGNED_DATA_DIR, const.TARGET_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    opt = Adam(mo.parameters(), lr=0.0002, betas=(0.5, 0.999))
    crt = nn.CrossEntropyLoss() if out_channels > 1 else nn.BCEWithLogitsLoss()

    for epoch in range(10):
        train_single_epoch(mo, dataloader, opt, crt)
