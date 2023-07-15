import os
import torch.nn as nn
import time
from itertools import product

from data_preparation.datasets.dataset_v3 import SdssDatasetV3
from trainer import Trainer
import const
import utils


def get_trainer(device, epochs, learning_rate, batch_size):
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
        batch_size=batch_size,
        learning_rate=learning_rate,
        adam_betas=const.ADAM_BETAS,
        device=device,
        epochs=epochs,
        criterion=crt,
        start_from_epoch=0,
        model_name=f"{learning_rate};{batch_size}",
        load_model_from=""
    )

    return trainer


def main():
    device = utils.get_device()
    utils.seed_torch()

    epochs = const.VALIDATION_EPOCHS
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [32, 128, 256]

    results = []
    for lr, bs in product(learning_rates, batch_sizes):
        trainer = get_trainer(device=device, epochs=epochs, learning_rate=lr, batch_size=bs)

        start_time = time.time()
        loss = trainer.train()
        total_time = time.time() - start_time

        result = (lr, bs, total_time, loss.detach())
        print(f"Result: {result}")

        results.append(result)
    minimum = min(results, key=lambda t: t[3])
    for r in results:
        print(r)

    print(f"Best result: {minimum}")


if __name__ == "__main__":
    main()
