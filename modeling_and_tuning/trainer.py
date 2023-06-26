import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb
import numpy as np
import os
from PIL import Image

import utils
import const
from metrics.metrics import *
from data_preparation.datasets.dataset_v3 import SdssDatasetV3
from inference import inference
from models.unet_v2 import UNetV2Smaller, UNetV2


class Trainer:
    def __init__(
            self,
            # model parameters
            model_input_channels,
            model_output_channels,
            model_bilinear,
            # dataset/loader parameters
            data_csv_path,
            batch_size,
            # optimizer parameters
            learning_rate,
            adam_betas,
            # general parameters
            device,
            epochs,
            criterion,
            start_from_epoch,
            load_model_from,
            evaluate_model_interval=const.EVALUATE_MODEL_INTERVAL,
            save_model_interval=const.SAVE_MODEL_INTERVAL,
            log_loss_iteration_interval=const.LOG_LOSS_ITERATION_INTERVAL
    ):
        self.device = device
        self.load_model_from = load_model_from

        # self.model = UNetV2Smaller(
        self.model = UNetV2(
            in_channels=model_input_channels,
            out_channels=model_output_channels,
            bilinear=model_bilinear
        ).to(self.device)

        self.epochs = epochs
        if start_from_epoch > 0:
            self._load_model()
            self.start_from_epoch = start_from_epoch
            print(f"Loading from {self.load_model_from}...")
        else:
            self.start_from_epoch = 0
            print("Starting training from scratch...")

        self.dataset = SdssDatasetV3(data_csv_path)
        self.dataset_size = len(self.dataset)
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        self.optimizer = Adam(
            self.model.parameters(), lr=learning_rate, betas=adam_betas
        )
        self.criterion = criterion

        wandb.login()
        wandb.init(project="AMLS", entity="luizz")

        self.evaluate_model_interval = evaluate_model_interval
        self.save_model_interval = save_model_interval
        self.log_loss_iteration_interval = log_loss_iteration_interval

    def train(self):
        for epoch in range(self.start_from_epoch, self.epochs):
            print(f"Epoch: {epoch + 1}")
            self._train_single_epoch(epoch)

            if (epoch + 1) % self.evaluate_model_interval == 0:
                print(f"Evaluating model after epoch {epoch + 1}... ", end='')
                self._evaluate(epoch + 1)
                print("Done")

            if (epoch + 1) % self.save_model_interval == 0:
                print(f"Saving model after epoch {epoch + 1}... ", end='')
                self._checkpoint(epoch)
                print("Done")

        print(f"Finished training, saving model... ", end='')
        self._checkpoint(self.epochs)
        print("Done")

    def _train_single_epoch(self, epoch):
        self.model.train()
        for i, (input_tensor, target_tensor) in enumerate(self.dataloader):
            iteration = (self.dataset_size // self.batch_size) * epoch + i
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            prediction_tensor = self.model(input_tensor)

            loss = self.criterion(prediction_tensor, target_tensor)
            if iteration % self.log_loss_iteration_interval == 0:
                print(f"\tLogging on {iteration} iteration... loss={loss}")
                wandb.log({"loss": loss}, step=iteration)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _evaluate(self, epoch):
        self.model.eval()

        _, target_img, result_img = inference(self.model)

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
        }, step=(self.dataset_size // self.batch_size) * epoch
        )

        # TODO: fix this and uncomment
        # if (epoch + 1) % const.SAVE_VALIDATION_RESULT_IMG_INTERVAL == 0:
        #     img = np.maximum(0, result_img)
        #     img = np.power(img, 0.5)
        #     img = Image.fromarray(img.astype(np.uint8))
        #     img.save(os.path.join(const.VALIDATION_OUTPUT_DIR, f"result-{epoch + 1}epoch.jpeg"))

    def _checkpoint(self, epoch):
        model_name = f"model-epoch={epoch+1}.pt"

        torch.save(self.model.state_dict(), model_name)

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.load_model_from))
        self.model.to(self.device)
