import pandas as pd
from torch.utils.data import Dataset
import re
import numpy as np
import torch

import utils
import const


class SdssDatasetV3(Dataset):
    def __init__(self, csv_name, transform=None):
        self.df = pd.read_csv(csv_name)
        self.size = len(self.df.index)
        self.transform = transform

    def __len__(self):
        if self.transform:
            return self.size * 2
        else:
            return self.size

    def __getitem__(self, idx):
        if idx > self.size:
            idx = self.size - idx

        row = self.df.iloc[idx]
        input_f = row[const.CSV_INPUT_COL]
        target_f = row[const.CSV_TARGET_COL]

        input_data = np.load(input_f)
        target_data = np.load(target_f)

        input_data = torch.from_numpy(input_data)
        target_data = torch.from_numpy(target_data)

        if self.transform:
            input_data = self.transform(input_data)
            target_data = self.transform(target_data)

        input_data = self._reshape(input_data)
        target_data = self._reshape(target_data)

        return input_data, target_data

    @staticmethod
    def _reshape(tensor):
        (H, W, Ch) = tensor.shape
        return torch.reshape(tensor, (Ch, H, W))