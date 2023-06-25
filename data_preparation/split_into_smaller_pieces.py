import numpy as np
import re
import os
import pandas as pd

import const
import utils


def split_files_and_save(files, output_dir, df, is_target):
    utils.create_dir_if_doesnt_exist(output_dir)
    created_files = []
    for f in files:
        img = np.load(f)
        pieces = utils.split_into_smaller_pieces(img)

        img_id = re.search(const.IMG_ID_REGEX, f).group()
        for idx, piece in enumerate(pieces):
            piece_file_name = f"{img_id}_{idx}_target.npy" if is_target else f"{img_id}_{idx}.npy"
            path = os.path.join(output_dir, piece_file_name)
            created_files.append(path)
            np.save(path, piece, allow_pickle=True)
    if is_target:
        df[const.CSV_TARGET_COL] = created_files
    else:
        df[const.CSV_INPUT_COL] = created_files


def split_images_in_directory(data_directory, inputs_directory, target_directory):
    input_files = utils.listdir_fullpath(inputs_directory)
    target_files = utils.listdir_fullpath(target_directory)
    target_files = list(filter(lambda f: "_target" in f, target_files))

    input_files = list(sorted(list(filter(lambda f: re.search(const.IMG_ID_REGEX, f), input_files))))
    target_files = list(sorted(list(filter(lambda f: re.search(const.IMG_ID_REGEX, f), target_files))))

    input_pieces_dir = os.path.join(data_directory, const.PIECE_DIR_INPUT_NAME)
    target_pieces_dir = os.path.join(data_directory, const.PIECE_DIR_TARGET_NAME)

    df = pd.DataFrame()
    split_files_and_save(input_files, input_pieces_dir, df, is_target=False)
    split_files_and_save(target_files, target_pieces_dir, df, is_target=True)
    df.to_csv(os.path.join(data_directory, const.CSV_NAME))


if __name__ == "__main__":
    split_images_in_directory(const.TRAIN_DIR, const.TRAIN_INPUTS_DIR, const.TRAIN_TARGETS_DIR)
    print("Train done...")
    split_images_in_directory(const.TEST_DIR, const.TEST_INPUTS_DIR, const.TEST_TARGETS_DIR)
    print("Test done...")
    split_images_in_directory(const.TRAIN_DIR, const.TRAIN_INPUTS_DIR, const.TRAIN_TARGETS_DIR)
    print("Validation done...")
