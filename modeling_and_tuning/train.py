from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch
import wandb

from data_preparation.dataset import SdssDataset
from models.unet import UNet
from criterions.dice_loss import DiceLoss
import utils
import const

device = utils.get_device()


def train_single_epoch(model, loader, optimizer, criterion):
    for i, (input_tensor, target_tensor) in enumerate(loader):
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        prediction_tensor = model(input_tensor)

        loss = criterion(prediction_tensor, target_tensor)

        wandb.log(
            {"loss": loss}
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    wandb.login()
    wandb.init(project="AMLS", entity="luizz")

    model_name = "model.pt"
    mo = UNet(in_channels=const.INPUT_CHANNELS, out_channels=const.OUTPUT_CHANNELS, bilinear=const.BILINEAR).to(device)

    dataset = SdssDataset(const.PIECES_READY_DATA_INPUTS_DIR, const.PIECES_READY_DATA_TARGETS_DIR)
    dataloader = DataLoader(dataset, batch_size=const.BATCH_SIZE, shuffle=True)
    opt = Adam(mo.parameters(), lr=const.LEARNING_RATE, betas=const.ADAM_BETAS)

    # crt = nn.CrossEntropyLoss() if const.OUTPUT_CHANNELS > 1 else nn.BCEWithLogitsLoss()
    crt = DiceLoss()

    for epoch in range(const.NUMBER_OF_EPOCHS):
        print(f"epoch={epoch + 1}")
        train_single_epoch(mo, dataloader, opt, crt)
        if (epoch + 1) % const.SAVE_MODEL_INTERVAL == 0:
            print("Saving model...")
            torch.save(mo.state_dict(), f"{model_name}-checkpoint-epoch={epoch+1}")
    torch.save(mo.state_dict(), model_name)
