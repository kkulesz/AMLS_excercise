import os
import wget
import logging

import const

BASE_URL = "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301"
BASE_CORDS_URL = "https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj"
LOG_FILE = "log.txt"

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s:\t%(message)s',
    datefmt='%H:%M:%S'
)

logging.info("Running Urban Planning")

logger = logging.getLogger('urbanGUI')


def build_data_url(run_num, filter_t, camera_col, sequence_num, f_ext):
    return f"{BASE_URL}/{run_num}/{camera_col}/frame-{filter_t}-{str(run_num).zfill(6)}-{camera_col}-{str(sequence_num).zfill(4)}.{f_ext}"


def build_cords_url(run_num, camera_col, cords_of_what):
    return f"{BASE_CORDS_URL}-{str(run_num).zfill(6)}-{camera_col}-{cords_of_what}.fits.gz"


def download_or_log_on_fail(url, dir):
    try:
        wget.download(url, dir)
    except:
        logger.log(logging.ERROR, url)


if __name__ == "__main__":
    data_dir = const.DATA_DIR

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # if len(os.listdir(data_dir)) != 0:
    #     print(f"'{data_dir}' directory is not empty, data is already downloaded probably - aborting!")
    #     exit(1)

    # list of tuples to easily modify urls
    # format: (<run_num>, <min_seq_number>, <max_seq_number>)
    # eg. (8162, 80, 237)
    # will give results:
    #   https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/3/frame-r-008162-1-0080.fits.bz2
    #   https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/3/frame-r-008162-1-0081.fits.bz2
    #   https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/3/frame-r-008162-1-0082.fits.bz2
    #   and so on...
    urls_meta_numbers = [
        (8162, 80, 81),  # 80, 237
        # (8110, 11, 225)
    ]

    for (run_num, min_seq, max_seq) in urls_meta_numbers:
        for camera_col in range(1, 7):

            # download coordinate files
            for cords_of_what in ["gal", "star"]:  # 2 * ~35MB = ~~70MB
                url = build_cords_url(run_num, camera_col, cords_of_what)
                download_or_log_on_fail(url, data_dir)
            print(f"Cords downloaded: {run_num}:{camera_col}")

            for seq_num in range(min_seq, max_seq):
                # download various bands of single image
                for band in ["g", "i", "r", "u", "z"]:  # 5 * ~3.5MB = ~~18MB
                    url = build_data_url(run_num, band, camera_col, seq_num, "fits.bz2")
                    download_or_log_on_fail(url, data_dir)
                print(f"Bands downloaded: {run_num}:{seq_num}:{camera_col}")

                # download irg jpgs
                for img_type in ["irg", "thumb-irg"]:  # less than 1MB combined, negligible,
                    url = build_data_url(run_num, img_type, camera_col, seq_num, "jpg")
                    download_or_log_on_fail(url, data_dir)
