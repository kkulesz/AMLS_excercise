from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import re
import pandas as pd
import os

import utils
import const


def process_coords(coords_fits, bands_fits, obj_type, coords_dir):
    coords_file = fits.open(coords_fits)
    raw_table = Table(coords_file[1].data)

    # TODO: change it later, because it is not optimal
    for b_f in bands_fits:
        band_file = fits.open(b_f)
        band_header = band_file[0].header
        band_wcs = WCS(band_header)
        rows, cols = band_file[0].data.shape
        obj_array = []
        for row in raw_table:
            obj_ra = row['RA']
            obj_dec = row['DEC']
            obj_skycoord = SkyCoord(obj_ra, obj_dec, unit='deg')
            xp, yp = obj_skycoord.to_pixel(band_wcs)
            if (0 <= xp <= cols) and (0 <= yp <= rows):  # check if object fits the image
                obj_array.append([xp, yp])
        result_filename = re.search(const.IMG_ID_REGEX, b_f).group()
        print(f"Saving {result_filename}_{obj_type}.csv...")
        pd.DataFrame(obj_array).to_csv(
            os.path.join(coords_dir, f"{result_filename}_{obj_type}.csv"),
            index=False
        )

        band_file.close()

    coords_file.close()
    return 0


def filter_files(files):
    r_band_regex = "r-[0-9]{6}-[1-6]-[0-9]{4}.fits$"
    band_fits_files = list(filter(lambda file_name: re.search(r_band_regex, file_name), files))

    gal_fits_files = list(filter(lambda file_name: file_name.endswith("gal.fits"), files))
    star_fits_files = list(filter(lambda file_name: file_name.endswith("star.fits"), files))

    return band_fits_files, gal_fits_files, star_fits_files


def assign_object_coords_to_bands(object_coords_fits_files, bands_fits_files):
    obj_bands_dict = {}
    for obj in object_coords_fits_files:
        run_col_regex = "[0-9]{6}-[1-6]"
        run_col_of_gal = re.search(run_col_regex, obj).group()

        bands = list(filter(lambda fb: re.search(run_col_of_gal, fb), bands_fits_files))
        obj_bands_dict[obj] = bands
    return obj_bands_dict


def process_everything_needed(coords_dir, fits_files_dir):
    files = utils.listdir_fullpath(fits_files_dir)
    band_fits_files, gal_fits_files, star_fits_files = filter_files(files)

    gal_bands_dict = assign_object_coords_to_bands(gal_fits_files, band_fits_files)
    star_bands_dict = assign_object_coords_to_bands(star_fits_files, band_fits_files)

    for item in gal_bands_dict.items():
        gal, bands = item
        process_coords(gal, bands, "gals", coords_dir)

    for item in star_bands_dict.items():
        star, bands = item
        process_coords(star, bands, "stars", coords_dir)


def main():
    coords_dir = const.COORDS_DATA_DIR
    data_dir = const.DATA_DIR
    utils.create_dir_if_doesnt_exist(coords_dir)

    process_everything_needed(coords_dir, data_dir)


if __name__ == "__main__":
    main()
