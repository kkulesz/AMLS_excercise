import numpy as np
import re
import os

import const
import utils


def get_multiple_without_reminder(N, divider):
    reminder = N % divider
    return N - reminder


def split_into_smaller_pieces(img):
    H, W, _ = img.shape
    nH, nW = const.PIECE_SHAPE
    mH = get_multiple_without_reminder(H, nH)
    mW = get_multiple_without_reminder(W, nW)
    return [img[x:x + nH, y:y + nW] for x in range(0, mH, nH) for y in range(0, mW, nW)]


def reconstruct_into_whole_image(pieces):
    H, W = const.ORIGINAL_IMAGE_SHAPE
    nH, nW = const.PIECE_SHAPE
    mW = get_multiple_without_reminder(W, nW)
    pieces_per_row = int(mW / nW)

    cols = [pieces[i:i + pieces_per_row] for i in range(0, len(pieces), pieces_per_row)]
    rows = []
    for col in cols:
        rows.append(np.concatenate(col, axis=1))
    print(rows[0].shape)
    return np.concatenate(rows, axis=0)


def split_images_in_directory(files, is_target, directory):
    for f in files:
        img = np.load(f)
        pieces = split_into_smaller_pieces(img)

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
