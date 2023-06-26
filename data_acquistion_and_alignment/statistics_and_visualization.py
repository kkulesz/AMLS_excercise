"""
TODO:
    Subsequently, compute meaningful summary statistics and visualization
    (e.g., a comparison of aligned and unaligned images).

    meaning:
        2. distribution of gals/stars:
            - in all downloaded data
            - in runs
            - in columns?
"""
from astropy.io import fits
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random

import const
import utils


def open_fits_and_return_data(band_path):
    band_file = fits.open(band_path)
    band_data = band_file[0].data
    band_file.close()
    return band_data


def save_aligned_and_not(image_num):
    unaligned_dir = const.DATA_DIR
    aligned_dir = const.ALIGNED_DATA_DIR

    g_fits, i_fits, r_fits, _, _ = tuple(
        sorted(list(filter(lambda f: image_num in f, utils.listdir_fullpath(unaligned_dir))))
    )
    aligned_file = list(filter(lambda f: image_num in f, utils.listdir_fullpath(aligned_dir)))[0]

    g_band = open_fits_and_return_data(g_fits)
    i_band = open_fits_and_return_data(i_fits)
    r_band = open_fits_and_return_data(r_fits)
    irg_unaligned = np.dstack((i_band, r_band, g_band))
    irg_aligned = np.load(aligned_file)

    s = 500
    irg_aligned = irg_aligned[:s, :s, :]
    irg_unaligned = irg_unaligned[:s, :s, :]

    # utils.display_image(irg_unaligned)
    # utils.display_image(irg_aligned)

    utils.save_image(irg_aligned, "aligned.jpg")
    utils.save_image(irg_unaligned, "unaligned.jpg")


def main():
    save_aligned_and_not("008162-6-0083")
    # save_aligned_and_not("003918-3-0213")


if __name__ == "__main__":
    main()
