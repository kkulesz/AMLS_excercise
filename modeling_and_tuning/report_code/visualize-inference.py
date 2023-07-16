import torch
import os

import const
import utils
from modeling_and_tuning.inference import inference
from modeling_and_tuning.models.unet_v2 import UNetV2Smaller


def main():
    device = utils.get_device()

    tuned = UNetV2Smaller(const.INPUT_CHANNELS, const.OUTPUT_CHANNELS)
    not_tuned = UNetV2Smaller(const.INPUT_CHANNELS, const.OUTPUT_CHANNELS)

    epoch = 80
    tuned.load_state_dict(torch.load(f"../../models_storage/tuned-smaller/tuned-{epoch}epochs.pt"))
    not_tuned.load_state_dict(torch.load(f"../../models_storage/not-tuned-smaller/not-tuned-{epoch}epochs.pt"))

    tuned.to(device)
    not_tuned.to(device)

    image_file = "008162-6-0083"
    input_path = os.path.join(const.TEST_INPUTS_DIR, f"{image_file}.npy")
    target_path = os.path.join(const.TEST_TARGETS_DIR, f"{image_file}_target.npy")

    raw_input_img, target_img, tuned_result = inference(
        tuned, img_size=(512, 512), input_path=input_path, target_path=target_path, random=False)

    _, _, not_tuned_result = inference(
        not_tuned, img_size=(512, 512), input_path=input_path, target_path=target_path, random=False)

    inference_dir = "artifacts/inference"
    utils.create_dir_if_doesnt_exist(inference_dir)

    utils.save_image(raw_input_img, f"{inference_dir}/input.jpeg", dpi=600)
    utils.save_image(target_img, f"{inference_dir}/target.jpeg", dpi=600)
    utils.save_image(tuned_result, f"{inference_dir}/tuned-{epoch}epoch-result.jpeg", dpi=600)
    utils.save_image(not_tuned_result, f"{inference_dir}/not-tuned-{epoch}epoch-result.jpeg", dpi=600)

    utils.display_image(raw_input_img)
    utils.display_image(target_img)
    utils.display_image(tuned_result)
    utils.display_image(not_tuned_result)


if __name__ == "__main__":
    main()
