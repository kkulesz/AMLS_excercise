import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import cv2
import re
import os

import utils
import const


def align_single(ref_header, band_path):
    ref_wcs = WCS(ref_header)
    ref_ra = ref_header['CRVAL1']
    ref_dec = ref_header['CRVAL2']
    ref_skycoord = SkyCoord(ref_ra, ref_dec, unit='deg')

    band_file = fits.open(band_path)
    band_header = band_file[0].header
    band_wcs = WCS(band_header)
    band_data = band_file[0].data

    num_rows, num_cols = band_data.shape[:2]

    band_ra = band_header['CRVAL1']
    band_dec = band_header['CRVAL2']
    band_skycoord = SkyCoord(band_ra, band_dec, unit='deg')

    other_image_pixel_coords = band_skycoord.to_pixel(ref_wcs)
    other_x = int(other_image_pixel_coords[0])
    other_y = int(other_image_pixel_coords[1])
    ref_x = ref_header['CRPIX1']
    ref_y = ref_header['CRPIX2']
    x_shift = other_x - ref_x
    y_shift = other_y - ref_y

    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    img = np.float32(band_data)

    return cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))


def align_spectral_bands(list_of_bands):
    sorted_list_of_bands = sorted(list_of_bands)  # make sure it is sorted
    if len(sorted_list_of_bands) != 5:
        raise "number of bands does not equal 5!"

    g, i, r, u, z = tuple(sorted_list_of_bands)

    ref_file = fits.open(r)
    ref_header = ref_file[0].header
    ref_wcs = WCS(ref_header)
    ref_data = ref_file[0].data

    rotated_g = align_single(ref_header, g)
    # rotated_r = align_single(ref_wcs, r)
    rotated_u = align_single(ref_header, u)
    rotated_i = align_single(ref_header, i)
    rotated_z = align_single(ref_header, z)

    ref_file.close()

    result = np.dstack((rotated_i, ref_data, rotated_g, rotated_u, rotated_z))

    # ##### PLOTTING CODE
    # img = result[:, :, :3]  # take irg channels for plotting
    # img = np.maximum(0, img)
    # img = np.power(img, 0.5)  # square root to make the high value pixels less dominant
    # plt.figure()
    # plt.axis("off")
    # plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    # plt.imshow(img)
    # plt.show()

    return result


if __name__ == "__main__":
    files = utils.listdir_fullpath(const.DATA_DIR)
    fits_files = list(filter(lambda file_name: file_name.endswith(".fits"), files))

    # grouping bands of the same image
    regex = "[0-9]{6}-[1-6]-[0-9]{4}"
    grouping_dict = {}
    for f in fits_files:
        img_id = re.search(regex, f).group()
        if img_id in grouping_dict:
            grouping_dict[img_id].append(f)
        else:
            grouping_dict[img_id] = [f]

    # aligning bands of the same image
    for item in grouping_dict.items():
        image_id, bands = item
        aligned_bands = align_spectral_bands(bands)

        path = os.path.join(const.DATA_DIR, "aligned", f"{image_id}")
        np.save(path, aligned_bands, allow_pickle=True)

