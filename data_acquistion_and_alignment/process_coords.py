from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import re
import numpy as np
import pandas as pd
import os

import utils
import const


def process_coords(coords_fits, bands_fits, obj_type):
    coords_file = fits.open(coords_fits)
    raw_table = Table(coords_file[1].data)

    # informative_columns = ['RUN', 'RERUN', 'CAMCOL', 'RA', 'DEC']
    # coords_data = fits. \
    #     BinTableHDU. \
    #     from_columns(
    #     [fits.Column(name=c, format=raw[c].dtype, array=raw[c]) for c in informative_columns]
    # ).data
    # filtered_data = raw[(raw['RUN'] == 3918) & (raw['CAMCOL'] == 1)]

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
            if (xp >= 0 and xp <= cols) and (yp >= 0 and yp <= rows):  # check if object fits the image
                obj_array.append([xp, yp])
        result_filename = re.search(const.IMG_ID_REGEX, b_f).group()
        print(f"Saving {result_filename}_{obj_type}.csv...")
        pd.DataFrame(obj_array).to_csv(os.path.join(const.COORDS_DATA_DIR, f"{result_filename}_{obj_type}.csv"))

        band_file.close()

    coords_file.close()
    return 0


if __name__ == "__main__":
    files = utils.listdir_fullpath(const.DATA_DIR)

    r_band_regex = "r-[0-9]{6}-[1-6]-[0-9]{4}.fits$"
    band_fits_files = list(filter(lambda file_name: re.search(r_band_regex, file_name), files))

    gal_fits_files = list(filter(lambda file_name: file_name.endswith("gal.fits"), files))
    star_fits_files = list(filter(lambda file_name: file_name.endswith("star.fits"), files))

    gal_bands_dict = {}
    for g in gal_fits_files:
        run_col_regex = "[0-9]{6}-[1-6]"
        run_col_of_gal = re.search(run_col_regex, g).group()

        bands = list(filter(lambda fb: re.search(run_col_of_gal, fb), band_fits_files))
        gal_bands_dict[g] = bands

    star_bands_dict = {}
    for s in star_fits_files:
        run_col_regex = "[0-9]{6}-[1-6]"
        run_col_of_stars = re.search(run_col_regex, s).group()

        bands = list(filter(lambda fb: re.search(run_col_of_stars, fb), band_fits_files))
        star_bands_dict[s] = bands

    for item in gal_bands_dict.items():
        gal, bands = item
        process_coords(gal, bands, "gals")

    for item in star_bands_dict.items():
        star, bands = item
        process_coords(star, bands, "stars")
