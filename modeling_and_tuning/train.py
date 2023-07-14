import os
import torch.nn as nn
import time

from data_preparation.datasets.dataset_v3 import SdssDatasetV3
from trainer import Trainer
import const
import utils

if __name__ == "__main__":
    device = utils.get_device()
    utils.seed_torch()

    train_csv_path = os.path.join(const.TRAIN_DIR, const.CSV_NAME)
    test_csv_path = os.path.join(const.TEST_DIR, const.CSV_NAME)
    validation_csv_path = os.path.join(const.VALIDATION_DIR, const.CSV_NAME)

    train_dataset = SdssDatasetV3(train_csv_path)
    test_dataset = SdssDatasetV3(test_csv_path)
    validation_dataset = SdssDatasetV3(validation_csv_path)
    crt = nn.CrossEntropyLoss() if const.OUTPUT_CHANNELS > 1 else nn.BCEWithLogitsLoss()

    trainer = Trainer(
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
        load_model_from=f"../models_storage/tuned-smaller/tuned-model-smaller-150epochs.pt",
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    print(f"Trained {const.NUMBER_OF_EPOCHS - const.START_EPOCH_FROM} epochs in {training_time}")
