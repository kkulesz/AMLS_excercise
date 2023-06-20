import torch.nn as nn
import torch
import numpy as np
import os
import cv2
import random

import utils
import const
from models.unet import UNet

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"


def validate_model_manual(
        model: nn.Module,
        img_size: int = None,
        input_path: str = const.VALIDATE_INPUT_PATH,
        target_path: str = const.VALIDATE_TARGET_PATH
):
    with torch.no_grad():
        raw_input = np.load(input_path)
        target_img = np.load(target_path)

        if img_size:
            raw_input = raw_input[0:img_size, 0:img_size, :]
            target_img = target_img[0:img_size, 0:img_size, :]

        img_H, img_W, _ = raw_input.shape
        pieces = utils.split_into_smaller_pieces(raw_input)
        pH, pW, pCh = pieces[0].shape
        pieces_transformed = []
        for piece in pieces:
            piece = np.reshape(piece, (pCh, pH, pW))
            piece = torch.from_numpy(piece)
            pieces_transformed.append(piece)
        input_tensor = torch.stack(pieces_transformed, dim=0)

        batches = []
        for i in range(0, len(input_tensor), const.BATCH_SIZE):
            batches.append(input_tensor[i: min(i + const.BATCH_SIZE, len(input_tensor))])

        results = []
        for batch in batches:
            b_result = model(batch.cuda())
            b_result = b_result.cpu().numpy()
            rB, rCh, rH, rW = b_result.shape
            b_result = np.reshape(b_result, (rB, rH, rW, rCh))
            results.append(b_result)
        result = np.concatenate(results, axis=0)
        result = utils.reconstruct_into_whole_image(result, img_H, img_W)
        # filtered = filter_image(result)

        utils.display_image(raw_input)
        utils.display_image(target_img)
        utils.display_image(result)
        # utils.display_image(filtered)


if __name__ == "__main__":
    model = UNet(const.INPUT_CHANNELS, const.OUTPUT_CHANNELS, bilinear=const.BILINEAR)
    model.load_state_dict(torch.load("model.pt"))
    model.to(utils.get_device())

    validate_model_manual(model, img_size=200)
