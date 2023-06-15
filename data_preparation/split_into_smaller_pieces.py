import numpy as np
import re
import os

import const
import utils


def get_multiple_without_reminder(N, divider):
    reminder = N % divider
    return N - reminder


def split_into_smaller_pieces(files, is_target, directory):
    for f in files:
        img = np.load(f)
        H, W, _ = img.shape
        nH, nW = const.PIECE_SHAPE

        mH = get_multiple_without_reminder(H, nH)
        mW = get_multiple_without_reminder(W, nW)
        pieces = [img[x:x + nH, y:y + nW] for x in range(0, mH, nH) for y in range(0, mW, nW)]

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

    split_into_smaller_pieces(input_files, is_target=False, directory=const.PIECES_READY_DATA_INPUTS_DIR)
    split_into_smaller_pieces(target_files, is_target=True, directory=const.PIECES_READY_DATA_TARGETS_DIR)
