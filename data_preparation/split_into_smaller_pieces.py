import numpy as np
import re
import os

import const
import utils


def split_images_in_directory(files, is_target, directory):
    for f in files:
        img = np.load(f)
        pieces = utils.split_into_smaller_pieces(img)

        img_id = re.search(const.IMG_ID_REGEX, f).group()
        for idx, piece in enumerate(pieces):
            piece_file_name = f"{img_id}_{idx}_target" if is_target else f"{img_id}_{idx}"
            path = os.path.join(directory, piece_file_name)
            np.save(path, piece, allow_pickle=True)


if __name__ == "__main__":
    utils.create_dir_if_doesnt_exist(const.PIECES_READY_DATA_DIR)
    utils.create_dir_if_doesnt_exist(const.PIECES_READY_DATA_INPUTS_DIR)
    utils.create_dir_if_doesnt_exist(const.PIECES_READY_DATA_TARGETS_DIR)

    input_files = utils.listdir_fullpath(const.ALIGNED_DATA_DIR)
    target_files = utils.listdir_fullpath(const.TARGET_DATA_DIR)
    target_files = list(filter(lambda f: "_target" in f, target_files))

    split_images_in_directory(input_files, is_target=False, directory=const.PIECES_READY_DATA_INPUTS_DIR)
    split_images_in_directory(target_files, is_target=True, directory=const.PIECES_READY_DATA_TARGETS_DIR)
