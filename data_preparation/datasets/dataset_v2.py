import pandas as pd
from torch.utils.data import Dataset
import re
import numpy as np
import torch

import utils
import const


class SdssDatasetV2(Dataset):
    def __init__(self, input_dir, target_dir):
        input_list = utils.listdir_fullpath(input_dir)
        target_list = utils.listdir_fullpath(target_dir)
        target_list = list(filter(lambda f: "_target" in f, target_list))

        input_list = sorted(input_list)
        target_list = sorted(target_list)

        example_image = np.load(input_list[0])
        self.pieces_per_file = len(utils.split_into_smaller_pieces(example_image))
        self.files = list(zip(input_list, target_list))
        self.size = len(self.files) * self.pieces_per_file

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_number = idx // self.pieces_per_file
        piece_number = self.size % self.pieces_per_file
        input_f, target_f = self.files[img_number]

        input_data = np.load(input_f)
        target_data = np.load(target_f)

        input_data = utils.split_into_smaller_pieces(input_data)[piece_number]
        target_data = utils.split_into_smaller_pieces(target_data)[piece_number]

        input_data = torch.from_numpy(input_data)
        target_data = torch.from_numpy(target_data)

        input_data = self._reshape(input_data)
        target_data = self._reshape(target_data)

        return input_data, target_data

    @staticmethod
    def _reshape(tensor):
        (H, W, Ch) = tensor.shape
        return torch.reshape(tensor, (Ch, H, W))