from torch.utils.data import Dataset
import re
import numpy as np
import torch

import utils
import const


class SdssDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        input_list = utils.listdir_fullpath(input_dir)
        target_list = utils.listdir_fullpath(target_dir)
        target_list = list(filter(lambda f: "target" in f, target_list))

        self.data = []
        for input_f in input_list:
            img_id = re.search(const.PIECE_ID_REGEX, input_f).group()[:-1]  # drop last char which isn't digit
            target_pattern = f"{img_id}_target"
            targets_containing_piece_id = [g for g in target_list if target_pattern in g]
            target_f = targets_containing_piece_id[0]
            self.data.append((input_f, target_f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_f, target_f = self.data[idx]

        input_data = np.load(input_f)
        target_data = np.load(target_f)

        input_data = utils.take_smaller_patch(input_data)
        target_data = utils.take_smaller_patch(target_data)

        input_data = torch.from_numpy(input_data)
        target_data = torch.from_numpy(target_data)

        input_data = self._reshape(input_data)
        target_data = self._reshape(target_data)

        return input_data, target_data

    @staticmethod
    def _reshape(tensor):
        (H, W, Ch) = tensor.shape
        return torch.reshape(tensor, (Ch, H, W))
