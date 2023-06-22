import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

import const


def listdir_fullpath(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]


def create_dir_if_doesnt_exist(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def display_image(img):
    img = img[:, :, :3]  # take irg channels for plotting
    img = np.maximum(0, img)
    img = np.power(img, 0.5)  # square root to make the high value pixels less dominant
    plt.figure()
    plt.axis("off")
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    plt.imshow(img)
    plt.show()


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Working on {device} device.")

    return device


def seed_torch(seed=2137):
    #  taken from: https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _get_multiple_without_reminder(N, divider):
    reminder = N % divider
    return N - reminder


def split_into_smaller_pieces(img):
    H, W, _ = img.shape
    nH, nW = const.PIECE_SHAPE
    mH = _get_multiple_without_reminder(H, nH)
    mW = _get_multiple_without_reminder(W, nW)
    return [img[x:x + nH, y:y + nW] for x in range(0, mH, nH) for y in range(0, mW, nW)]


def reconstruct_into_whole_image(pieces, org_H, org_W):
    # H, W = const.ORIGINAL_IMAGE_SHAPE
    nH, nW = const.PIECE_SHAPE
    mW = _get_multiple_without_reminder(org_W, nW)
    pieces_per_row = int(mW / nW)

    cols = [pieces[i:i + pieces_per_row] for i in range(0, len(pieces), pieces_per_row)]
    rows = []
    for col in cols:
        rows.append(np.concatenate(col, axis=1))
    return np.concatenate(rows, axis=0)
