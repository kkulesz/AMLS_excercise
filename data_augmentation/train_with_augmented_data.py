import os
import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

import const
import utils
from data_preparation.datasets.dataset_v3 import SdssDatasetV3
from modeling_and_tuning.trainer import Trainer


def get_trainer(train_dataset, test_dataset, validation_dataset, device, name):
    crt = nn.CrossEntropyLoss() if const.OUTPUT_CHANNELS > 1 else nn.BCEWithLogitsLoss()
    return Trainer(
        model_input_channels=const.INPUT_CHANNELS,
        model_output_channels=const.OUTPUT_CHANNELS,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        validation_dataset=validation_dataset,
        batch_size=const.BATCH_SIZE,
        learning_rate=const.LEARNING_RATE,
        adam_betas=const.ADAM_BETAS,
        device=device,
        epochs=const.NUMBER_OF_EPOCHS,
        criterion=crt,
        start_from_epoch=const.START_EPOCH_FROM,
        load_model_from=const.LOAD_MODEL_FROM,
        model_name=name
    )


def main():
    device = utils.get_device()
    utils.seed_torch()

    transform = RandomVerticalFlip(p=const.TRANSFORM_PROBABILITY)
    name = transform.__class__.__name__

    train_csv_path = os.path.join(const.TRAIN_DIR, const.CSV_NAME)
    test_csv_path = os.path.join(const.TEST_DIR, const.CSV_NAME)
    validation_csv_path = os.path.join(const.VALIDATION_DIR, const.CSV_NAME)

    train_dataset = SdssDatasetV3(train_csv_path, transform=transform)
    test_dataset = SdssDatasetV3(test_csv_path)
    validation_dataset = SdssDatasetV3(validation_csv_path)

    trainer = get_trainer(train_dataset, test_dataset, validation_dataset, device, name)

    trainer.train()


if __name__ == "__main__":
    main()
