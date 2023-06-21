import numpy as np
import re
import os

import const
import utils


def split_files_and_save(files, output_dir, is_target):
    utils.create_dir_if_doesnt_exist(output_dir)
    for f in files:
        img = np.load(f)
        pieces = utils.split_into_smaller_pieces(img)

        img_id = re.search(const.IMG_ID_REGEX, f).group()
        for idx, piece in enumerate(pieces):
            piece_file_name = f"{img_id}_{idx}_target" if is_target else f"{img_id}_{idx}"
            path = os.path.join(output_dir, piece_file_name)
            np.save(path, piece, allow_pickle=True)


def split_images_in_directory(data_directory, target_directory):
    input_files = utils.listdir_fullpath(data_directory)
    target_files = utils.listdir_fullpath(target_directory)
    target_files = list(filter(lambda f: "_target" in f, target_files))

    input_files = list(filter(lambda f: re.search(const.IMG_ID_REGEX, f), input_files))
    target_files = list(filter(lambda f: re.search(const.IMG_ID_REGEX, f), target_files))

    input_pieces_dir = os.path.join(data_directory, const.PIECE_DIR_INPUT_NAME)
    target_pieces_dir = os.path.join(data_directory, const.PIECE_DIR_TARGET_NAME)

    split_files_and_save(input_files, input_pieces_dir, is_target=False)
    split_files_and_save(target_files, target_pieces_dir, is_target=False)


if __name__ == "__main__":
    target_name = "target"
    split_images_in_directory(const.TRAIN_DIR, os.path.join(const.TRAIN_DIR, target_name))
    split_images_in_directory(const.TEST_DIR, os.path.join(const.TEST_DIR, target_name))
    split_images_in_directory(const.VALIDATION_DIR, os.path.join(const.VALIDATION_DIR, target_name))
