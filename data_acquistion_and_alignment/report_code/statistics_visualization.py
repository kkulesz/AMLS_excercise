import matplotlib.pyplot as plt
import pandas as pd
import re
import statistics
import os

import utils
import const


def read_files_and_get_numbers(directory):
    cord_files = list(sorted(utils.listdir_fullpath(directory)))
    images_gal_star_files = [cord_files[i:i + 2] for i in range(0, len(cord_files), 2)]

    img_id_gals_stars_count = []
    for f in images_gal_star_files:
        gals_file = f[0]
        stars_file = f[1]

        img_id = re.search(const.IMG_ID_REGEX, gals_file).group()
        img_id_to_check = re.search(const.IMG_ID_REGEX, stars_file).group()
        assert img_id == img_id_to_check  # sanity check

        gal_count = pd.read_csv(gals_file).size
        star_count = pd.read_csv(stars_file).size
        img_id_gals_stars_count.append((img_id, gal_count, star_count))

    return img_id_gals_stars_count


def show_bar_plot_with_avg_line(X_list, Y_list, Y_label, X_label="Image-id"):
    avg_value = statistics.fmean(Y_list)

    plt.bar(X_list, Y_list)
    plt.ylabel(Y_label)
    plt.xlabel(X_label)
    plt.axhline(y=avg_value, linewidth=1, color='r')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.25)
    path = os.path.join("artifacts", f"{Y_label}.png")
    plt.savefig(path, dpi=1000)
    plt.show()


def process_numbers(img_id_gals_stars_count):
    gal_count_list = [g for _, g, _ in img_id_gals_stars_count]
    star_count_list = [s for _, _, s in img_id_gals_stars_count]
    image_ids_list = [img_id for img_id, _, _ in img_id_gals_stars_count]

    show_bar_plot_with_avg_line(X_list=image_ids_list, Y_list=gal_count_list, Y_label="Number-of-gals")
    show_bar_plot_with_avg_line(X_list=image_ids_list, Y_list=star_count_list, Y_label="Number-of-stars")

    gal_to_star_ratios = [g/s for _, g, s in img_id_gals_stars_count]
    show_bar_plot_with_avg_line(X_list=image_ids_list, Y_list=gal_to_star_ratios, Y_label="Number-of-gals-to-stars-ratio")


def main():
    img_id_gals_stars_count = read_files_and_get_numbers(const.COORDS_DATA_DIR)
    process_numbers(img_id_gals_stars_count)


if __name__ == "__main__":
    main()
