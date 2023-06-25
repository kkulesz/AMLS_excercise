import os
import torch.nn as nn

from trainer import Trainer
import const
import utils

if __name__ == "__main__":
    device = utils.get_device()
    csv_path = os.path.join(const.TRAIN_DIR, const.CSV_NAME)
    crt = nn.CrossEntropyLoss() if const.OUTPUT_CHANNELS > 1 else nn.BCEWithLogitsLoss()

    trainer = Trainer(
        model_input_channels=const.INPUT_CHANNELS,
        model_output_channels=const.OUTPUT_CHANNELS,
        model_bilinear=const.BILINEAR,
        data_csv_path=csv_path,
        batch_size=const.BATCH_SIZE,
        learning_rate=const.LEARNING_RATE,
        adam_betas=const.ADAM_BETAS,
        device=device,
        epochs=const.NUMBER_OF_EPOCHS,
        criterion=crt,
        start_from_epoch=const.START_EPOCH_FROM,
        load_model_from=f"model-epoch={const.START_EPOCH_FROM+1}.pt",
        # evaluate_model_interval=const.EVALUATE_MODEL_INTERVAL,
        # save_model_interval=const.SAVE_MODEL_INTERVAL,
        # log_loss_iteration_interval=const.LOG_LOSS_ITERATION_INTERVAL
    )

    trainer.train()
