import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from scipy.ndimage import shift, rotate

import utils
import const


def align_single(ref_header, band_path):
    ref_wcs = WCS(ref_header)

    curr_band_file = fits.open(band_path)
    curr_header = curr_band_file[0].header
    curr_wcs = WCS(curr_header)
    curr_data = curr_band_file[0].data
    ref_x, ref_y = curr_data.shape

    # x_offset = ref_header['CRPIX1'] - curr_header['CRPIX1'] # ZERO?
    # y_offset = ref_header['CRPIX2'] - curr_header['CRPIX2'] # ZERO?
    ##############################################################################
    target_ra = ref_header['CRVAL1']
    target_dec = ref_header['CRVAL2']

    target_coord = SkyCoord(target_ra, target_dec, unit='deg')
    cutout = Cutout2D(curr_data, target_coord, curr_data.shape, wcs=curr_wcs)
    cutout_header = cutout.wcs.to_header()

    res_x, res_y = cutout.data.shape
    off_x = ref_x - res_x
    off_y = ref_y - res_y
    # print(off_x)
    # print(off_y)

    # res = curr_data
    res = cutout.data

    # res = np.roll(cutout.data, (off_x, off_y), axis=(0, 1))
    # res = np.roll(cutout.data, (-off_x, -off_y), axis=(0, 1))

    # res = np.roll(cutout.data, (off_y, off_x), axis=(0, 1))
    # res = np.roll(cutout.data, (-off_y, -off_x), axis=(0, 1))
    ##############################################################################
    # ref_crpix1 = ref_header['CRPIX1']
    # ref_crpix2 = ref_header['CRPIX2']
    # curr_crpix1 = curr_header['CRPIX1']
    # curr_crpix2 = curr_header['CRPIX2']
    # # Convert pixel coordinates to world coordinates
    # ref_world = ref_wcs.pixel_to_world(ref_crpix1, ref_crpix2)
    # curr_world = curr_wcs.pixel_to_world(curr_crpix1, curr_crpix2)
    # # Calculate the offset in pixel coordinates
    # delta_pix = ref_wcs.world_to_pixel(curr_world) - np.array([ref_crpix1, ref_crpix2])
    # aligned_data = np.zeros_like(curr_data)
    # # Shift the current band data to align with the reference band
    # aligned_data += np.roll(curr_data, int(round(delta_pix[0])) - 1, axis=1)
    # aligned_data += np.roll(curr_data, int(round(delta_pix[1])) - 1, axis=0)
    ##############################################################################

    curr_band_file.close()

    # return aligned_data[500:1000, 500:1500]
    return res[100:1000, 100:1500]
    # return curr_data[100:1000, 100:1500]


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
    # rotated_u = align_single(ref_header, u)
    rotated_i = align_single(ref_header, i)
    # rotated_z = align_single(ref_wcs, z)

    ref_file.close()

    rgb_default = make_lupton_rgb(rotated_i, ref_data[100:1000, 100:1500], rotated_g, stretch=1.5, Q=10)
    plt.imshow(rgb_default, origin="lower")
    plt.show()

    # rgb_default = make_lupton_rgb(r_data, g_rotated_image, u_data, stretch=1.5, Q=10)
    # plt.imshow(rgb_default, origin="lower")
    # plt.show()

    # plt.savefig("base.jpg")
    # print(ref_wcs)
    # print(ref_wcs.pixel_to_world)
    # print(ref_wcs.world_to_pixel)
    # print(ref_wcs)

    # ref_band = fits.open(r)
    # ref_header = ref_band[0].header
    # print(ref_header['CRPIX1'])
    # print(ref_header['CRPIX2'])
    # print(ref_header['CRVAL1'])
    # print(ref_header['CRVAL2'])
    # print(ref_header['CD1_1'])
    # print(ref_header['CD1_2'])
    # print(ref_header['CD2_1'])
    # print(ref_header['CD2_2'])
    #
    # print("---"* 30)
    # ref_band = fits.open(g)
    # ref_header = ref_band[0].header
    # print(ref_header['CRPIX1'])
    # print(ref_header['CRPIX2'])
    # print(ref_header['CRVAL1'])
    # print(ref_header['CRVAL2'])
    # print(ref_header['CD1_1'])
    # print(ref_header['CD1_2'])
    # print(ref_header['CD2_1'])
    # print(ref_header['CD2_2'])


if __name__ == "__main__":
    files = utils.listdir_fullpath(const.DATA_DIR)
    fits_files = list(filter(lambda file_name: file_name.endswith(".fits"), files))
    # for f in fits_files:
    #     print(f)

    aligned_bands = align_spectral_bands(fits_files)
    # rgb_image = create_rgb_image(aligned_bands)
    # output_file = "aligned_image.jpg"
    # plt.imsave(output_file, rgb_image)
