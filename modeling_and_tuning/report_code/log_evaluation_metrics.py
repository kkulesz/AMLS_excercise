import torch.nn as nn
import torch
import numpy as np
import os
import re
import wandb
from torch.utils.data import DataLoader

import utils
import const
from data_preparation.datasets.dataset_v3 import SdssDatasetV3
from modeling_and_tuning.models.unet_v2 import UNetV2, UNetV2Smaller
from modeling_and_tuning.metrics.metrics import dice_coef, iou, accuracy, recall_score, precision_score


def get_models(models_dir, device):
    models_files = utils.listdir_fullpath(models_dir)
    models_files = list(sorted(models_files))

    models_with_epoch_num = []
    for f in models_files:
        epoch_piece = re.findall(r'\d+epoch', f)[0]
        epoch = int(re.search(r'\d+[^0-9]', epoch_piece).group()[:-1])
        model = UNetV2Smaller(const.INPUT_CHANNELS, const.OUTPUT_CHANNELS).to(device)
        model.load_state_dict(torch.load(f))

        models_with_epoch_num.append((model, epoch))
    models_with_epoch_num.sort(key=lambda t: t[1])
    return models_with_epoch_num


def calculate_and_log_for_single_model(model, loader, epoch, device, dataset_size):
    iou_score = 0
    dice_score = 0
    prec_score = 0
    acc_score = 0
    rec_score = 0
    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(loader):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            prediction_tensor = model(input_tensor)

            target_numpy = target_tensor.cpu().numpy()
            prediction_numpy = prediction_tensor.cpu().numpy()

            iou_score += iou(target_numpy, prediction_numpy)
            dice_score += dice_coef(target_numpy, prediction_numpy)
            prec_score += precision_score(target_numpy, prediction_numpy)
            acc_score += accuracy(target_numpy, prediction_numpy)
            rec_score += recall_score(target_numpy, prediction_numpy)

    loader_len = len(loader)

    iou_score = iou_score / loader_len
    dice_score = dice_score / loader_len
    prec_score = prec_score / loader_len
    acc_score = acc_score / loader_len
    rec_score = rec_score / loader_len

    print(f"{epoch}:"
          f"\t dice: {dice_score}"
          f"\t iou: {iou_score}"
          f"\t accuracy: {acc_score}"
          f"\t precision: {prec_score}"
          f"\t recall: {rec_score}")

    wandb.log(
        {
            "dice_coefficient": dice_score,
            "iou": iou_score,
            "accuracy": acc_score,
            "precision": prec_score,
            "recall": rec_score
        }, step=epoch
    )


def main():
    tuned = True
    if tuned:
        name = "tuned-metrics"
        dir_name = "tuned-smaller"
    else:
        name = "not-tuned-metrics"
        dir_name = "not-tuned-smaller"

    wandb.login()
    wandb.init(project="AMLS", entity="luizz", reinit=True, name=name)

    device = utils.get_device()
    models_dir = os.path.join(const.ABSOLUTE_PATH, "models_storage", dir_name)
    models_with_epoch_number = get_models(models_dir, device)

    test_csv_path = os.path.join(const.TEST_DIR, const.CSV_NAME)
    test_dataset = SdssDatasetV3(test_csv_path)
    test_dataloader = DataLoader(test_dataset, batch_size=const.BATCH_SIZE // 2, shuffle=False)

    for model, epoch in models_with_epoch_number:
        calculate_and_log_for_single_model(model, test_dataloader, epoch, device, len(test_dataset))


if __name__ == "__main__":
    main()
