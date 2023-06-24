from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch
import wandb
import os
from PIL import Image

from data_preparation.dataset import SdssDataset
from models.unet import UNet
from models.unet_v2 import UNetV2
from metrics.metrics import *
from inference import inference
import utils
import const

device = utils.get_device()


def train_single_epoch(model, loader, optimizer, criterion):
    model.train()
    for i, (input_tensor, target_tensor) in enumerate(loader):
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        prediction_tensor = model(input_tensor)

        loss = criterion(prediction_tensor, target_tensor)
        wandb.log({"loss": loss})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, epoch):
    model.eval()

    _, target_img, result_img = inference(model)

    target_img = utils.clip_target_to_output_shape(target_img, result_img)

    prec_score = precision_score(target_img, result_img)
    rec_score = recall_score(target_img, result_img)
    acc_score = accuracy(target_img, result_img)
    dice_score = dice_coef(target_img, result_img)
    iou_score = iou(target_img, result_img)

    utils.display_image(result_img)

    wandb.log({
        "precision": prec_score,
        "recall": rec_score,
        "accuracy": acc_score,
        "dice_coefficient": dice_score,
        "iou": iou_score,
    })

    if (epoch + 1) % const.SAVE_VALIDATION_RESULT_IMG_INTERVAL == 0:
        im = Image.fromarray((result_img * 255).astype(np.uint8))
        im.save(os.path.join(const.VALIDATION_OUTPUT_DIR, f"result-{epoch+1}epoch.jpeg"))


if __name__ == "__main__":
    utils.create_dir_if_doesnt_exist(const.VALIDATION_OUTPUT_DIR)

    wandb.login()
    wandb.init(project="AMLS", entity="luizz")

    start_from_epoch = const.START_EPOCH_FROM

    utils.seed_torch()
    inputs_dir = os.path.join(const.TRAIN_DIR, const.PIECE_DIR_INPUT_NAME)
    targets_dir = os.path.join(const.TRAIN_DIR, const.PIECE_DIR_TARGET_NAME)

    model_name = "model"
    # mo = UNet(in_channels=const.INPUT_CHANNELS, out_channels=const.OUTPUT_CHANNELS, bilinear=const.BILINEAR).to(device)
    mo = UNetV2(in_channels=const.INPUT_CHANNELS, out_channels=const.OUTPUT_CHANNELS, bilinear=const.BILINEAR).to(device)

    if start_from_epoch > 0:
        mo.load_state_dict(torch.load("../models_storage/model.pt"))
        mo.to(device)
        print(f"Loading model and starting from {start_from_epoch}. epoch.")
    else:
        print("Learning from scratch...")

    dataset = SdssDataset(inputs_dir, targets_dir)
    dataloader = DataLoader(dataset, batch_size=const.BATCH_SIZE, shuffle=True)
    opt = Adam(mo.parameters(), lr=const.LEARNING_RATE, betas=const.ADAM_BETAS)

    crt = nn.CrossEntropyLoss() if const.OUTPUT_CHANNELS > 1 else nn.BCEWithLogitsLoss()

    for epoch in range(start_from_epoch, const.NUMBER_OF_EPOCHS):
        print(f"epoch={epoch + 1}")
        train_single_epoch(mo, dataloader, opt, crt)
        if (epoch + 1) % const.EVALUATE_MODEL_INTERVAL == 0:
            evaluate(mo, epoch)

        if (epoch + 1) % const.SAVE_MODEL_INTERVAL == 0:
            print("Saving model...")
            torch.save(mo.state_dict(), f"{model_name}-checkpoint-epoch={epoch + 1}.pt")

    torch.save(mo.state_dict(), f"{model_name}.pt")
