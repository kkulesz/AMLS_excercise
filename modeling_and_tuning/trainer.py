import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb

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
            # dataset/loader parameters
            train_data_csv_path,
            test_data_csv_path,
            validation_data_csv_path,
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
            validate_model_interval=const.VALIDATE_MODEL_INTERVAL,
            test_model_interval=const.TEST_MODEL_INTERVAL,
            save_inference_result_img_interval=const.SAVE_INFERENCE_RESULT_IMG_INTERVAL,
            save_model_interval=const.SAVE_MODEL_INTERVAL,
            log_loss_iteration_interval=const.LOG_LOSS_ITERATION_INTERVAL
    ):
        self.device = device
        self.load_model_from = load_model_from

        self.model = UNetV2Smaller(
            # self.model = UNetV2(
            in_channels=model_input_channels,
            out_channels=model_output_channels
        ).to(self.device)

        self.epochs = epochs
        if start_from_epoch > 0:
            self._load_model()
            self.start_from_epoch = start_from_epoch
            print(f"Loading from {self.load_model_from}...")
        else:
            self.start_from_epoch = 0
            print("Starting training from scratch...")

        self.train_dataset = SdssDatasetV3(train_data_csv_path)
        self.test_dataset = SdssDatasetV3(test_data_csv_path)
        self.validation_dataset = SdssDatasetV3(validation_data_csv_path)

        self.train_dataset_size = len(self.train_dataset)
        self.batch_size = batch_size

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        # setting smaller batch size in test and validation dataloader because CudaOutOfMemory error
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size//2, shuffle=False)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size//2, shuffle=False)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, betas=adam_betas)
        self.criterion = criterion

        wandb.login()
        self.name = f"not-tuned-model-smaller"
        wandb.init(project="AMLS", entity="luizz", reinit=True, name=self.name)

        self.validate_model_interval = validate_model_interval
        self.test_model_interval = test_model_interval
        self.save_inference_result_img_interval = save_inference_result_img_interval
        self.save_model_interval = save_model_interval
        self.log_loss_iteration_interval = log_loss_iteration_interval

    def train(self):
        for epoch in range(self.start_from_epoch, self.epochs):
            print(f"Epoch: {epoch + 1}... ", end='')
            train_loss = self._train_single_epoch()
            print(f"training loss: {train_loss}")
            wandb.log({"training_loss": train_loss}, step=epoch+1)

            # if (epoch + 1) % self.validate_model_interval == 0:
            #     print(f"Evaluating on validation set after epoch number {epoch + 1}... ", end='')
            #     self._validate(epoch + 1)
            #     print("Done")

            if (epoch + 1) % self.test_model_interval == 0:
                print(f"Evaluating on test set after epoch number {epoch + 1}... ", end='')
                self._test(epoch + 1)
                print("Done")

            if (epoch + 1) % self.save_inference_result_img_interval == 0:
                print(f"Dumping result image after epoch {epoch + 1}... ", end='')
                self._dump_result_of_test_image_inference(epoch + 1)
                print("Done")

            if (epoch + 1) % self.save_model_interval == 0:
                print(f"Saving model after epoch {epoch + 1}... ", end='')
                self._checkpoint(epoch + 1)
                print("Done")

        print(f"Finished training, saving model... ", end='')
        self._checkpoint(self.epochs)
        print("Done")

        print(f"And running evaluation on test...", end='')
        final_test_loss = self._test(self.epochs)
        print("Done")

        return final_test_loss

    def _validate(self, epoch):
        return self._iterate_over_loader_and_log_loss(self.validation_dataloader, "validation_loss", epoch, len(self.validation_dataset))

    def _test(self, epoch):
        return self._iterate_over_loader_and_log_loss(self.test_dataloader, "test_loss", epoch, len(self.test_dataset))

    def _iterate_over_loader_and_log_loss(self, loader, log_label, epoch, dataset_size):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (input_tensor, target_tensor) in enumerate(loader):
                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                prediction_tensor = self.model(input_tensor)

                total_loss += self.criterion(prediction_tensor, target_tensor)

        loss = total_loss/dataset_size
        wandb.log({log_label: loss}, step=epoch)
        return loss

    def _train_single_epoch(self):
        self.model.train()
        total_loss = 0
        for i, (input_tensor, target_tensor) in enumerate(self.train_dataloader):
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            prediction_tensor = self.model(input_tensor)

            loss = self.criterion(prediction_tensor, target_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss
        return total_loss / self.train_dataset_size

    def _dump_result_of_test_image_inference(self, epoch):
        self.model.eval()

        _, _, result_img = inference(self.model)
        utils.save_image(result_img, f"result-{epoch}epoch.jpeg", dpi=600)

        # target_img = utils.clip_target_to_output_shape(target_img, result_img)
        #
        # prec_score = precision_score(target_img, result_img)
        # rec_score = recall_score(target_img, result_img)
        # acc_score = accuracy(target_img, result_img)
        # dice_score = dice_coef(target_img, result_img)
        # iou_score = iou(target_img, result_img)
        #
        # utils.display_image(result_img)
        #
        # wandb.log({
        #     "precision": prec_score,
        #     "recall": rec_score,
        #     "accuracy": acc_score,
        #     "dice_coefficient": dice_score,
        #     "iou": iou_score,
        # }, step=(self.train_dataset_size // self.batch_size) * epoch
        # )

    def _checkpoint(self, epoch):
        model_name = f"{self.name}-{epoch}epochs.pt"

        torch.save(self.model.state_dict(), model_name)

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.load_model_from))
        self.model.to(self.device)
