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
from modeling_and_tuning.report_code.log_evaluation_metrics import calculate_and_log_for_single_model


def main():
    # model_dir = "RandomHorizontalFlip"
    # model_dir = "RandomVerticalFlip"
    # model_name = f"{model_dir}-20epochs.pt"

    model_dir = "../../models_storage/tuned-smaller"
    model_name = "tuned-20epochs.pt"

    wandb.login()
    wandb.init(project="AMLS", entity="luizz", reinit=True, name=f"{model_name}-metrics")

    device = utils.get_device()
    model_path = os.path.join("models", model_dir, model_name)
    model = UNetV2Smaller(const.INPUT_CHANNELS, const.OUTPUT_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path))

    test_csv_path = os.path.join(const.TEST_DIR, const.CSV_NAME)
    test_dataset = SdssDatasetV3(test_csv_path)
    test_dataloader = DataLoader(test_dataset, batch_size=const.BATCH_SIZE // 2, shuffle=False)

    calculate_and_log_for_single_model(model, test_dataloader, 20, device, len(test_dataset))


if __name__ == "__main__":
    main()
