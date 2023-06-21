import os
import re
import numpy as np
import pandas as pd

import const
import utils


def color_pixels(img, df_of_pixels, channel):
    ps = 3
    for _, coords in df_of_pixels.iterrows():
        x = int(coords[0])
        y = int(coords[1])
        img[y - ps:y + ps, x - ps:x + ps, 2] = 0.
        img[y - ps:y + ps, x - ps:x + ps, channel] = 1.
    return img


def materialize_target(img_numpy_file, gal_csv, star_csv, target_dir):
    img = np.load(img_numpy_file)
    gal_df = pd.read_csv(gal_csv, skiprows=0)
    star_df = pd.read_csv(star_csv, skiprows=0)

    rows, cols, _ = img.shape

    target_image = np.empty((rows, cols, 3))
    target_image[:, :, 2] = np.full((rows, cols), 1.)

    target_image = color_pixels(target_image, gal_df, channel=0)
    target_image = color_pixels(target_image, star_df, channel=1)

    marked_image = img[:, :, :3]
    marked_image = color_pixels(marked_image, gal_df, channel=0)
    marked_image = color_pixels(marked_image, star_df, channel=1)

    # utils.display_image(target_image)
    # utils.display_image(marked_image)

    img_id = re.search(const.IMG_ID_REGEX, img_numpy_file).group()
    target_path = os.path.join(target_dir, f"{img_id}_target")
    marked_path = os.path.join(target_dir, f"{img_id}_marked")
    np.save(target_path, target_image, allow_pickle=True)
    np.save(marked_path, marked_image, allow_pickle=True)


def materialize_target_for_directory(data_dir, target_data_dir, coords_dir):
    utils.create_dir_if_doesnt_exist(target_data_dir)

    coords_files = utils.listdir_fullpath(coords_dir)
    gals_files = list(filter(lambda f: re.search("gal", f), coords_files))
    stars_files = list(filter(lambda f: re.search("star", f), coords_files))

    image_files = utils.listdir_fullpath(data_dir)
    image_files = list(filter(lambda f: re.search(const.IMG_ID_REGEX, f), image_files))

    img_gal_star_files_tuples = []
    for img_f in image_files:
        img_id = re.search(const.IMG_ID_REGEX, img_f).group()
        gal_f = next(g for g in gals_files if img_id in g)
        star_f = next(s for s in stars_files if img_id in s)

        img_gal_star_files_tuples.append(
            (img_f, gal_f, star_f)
        )

    for (img, gal, star) in img_gal_star_files_tuples:
        materialize_target(img, gal, star, target_data_dir)


if __name__ == "__main__":
    target_name = "target"
    materialize_target_for_directory(const.TRAIN_DIR, os.path.join(const.TRAIN_DIR, target_name), const.COORDS_DATA_DIR)
    materialize_target_for_directory(const.TEST_DIR, os.path.join(const.TEST_DIR, target_name), const.COORDS_DATA_DIR)
    materialize_target_for_directory(const.VALIDATION_DIR, os.path.join(const.VALIDATION_DIR, target_name), const.COORDS_DATA_DIR)

