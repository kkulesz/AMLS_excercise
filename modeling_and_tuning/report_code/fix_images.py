import torch.nn as nn
import torch
import numpy as np
import os
import re

import utils
import const
from modeling_and_tuning.models.unet_v2 import UNetV2, UNetV2Smaller
from modeling_and_tuning.inference import inference


def main():
    device = utils.get_device()
    models_dir = os.path.join(const.ABSOLUTE_PATH, "models_storage", "tuned-smaller")
    models_files = utils.listdir_fullpath(models_dir)
    models_files = list(sorted(models_files))

    for f in models_files:
        print(f)
        epoch_piece = re.findall(r'\d+epoch', f)[0]
        epoch = int(re.search(r'\d+[^0-9]', epoch_piece).group()[:-1])
        if epoch < 60:
            continue

        model = UNetV2Smaller(const.INPUT_CHANNELS, const.OUTPUT_CHANNELS).to(device)
        model.load_state_dict(torch.load(f))

        _, _, result_img = inference(model)
        utils.save_image(result_img, f"result-{epoch}epoch.jpeg", dpi=600)


if __name__ == '__main__':
    main()
