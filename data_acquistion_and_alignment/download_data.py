import os
import wget
import bz2
import logging

BASE_URL = "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301"
DATA_DIRECTORY = "../data/"
LOG_FILE = "log.txt"

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s:\t%(message)s',
    datefmt='%H:%M:%S'
)

logging.info("Running Urban Planning")

logger = logging.getLogger('urbanGUI')


def build_url(run_num, filter_t, camera_col, sequence_num, f_ext):
    return f"{BASE_URL}/{run_num}/{camera_col}/frame-{filter_t}-{str(run_num).zfill(6)}-{camera_col}-{str(sequence_num).zfill(4)}.{f_ext}"


def download_or_log_at_fail(url, dir):
    try:
        wget.download(url, dir)
    except:
        logger.log(logging.ERROR, url)


if __name__ == "__main__":

    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)

    if len(os.listdir(DATA_DIRECTORY)) != 0:
        print(f"'{DATA_DIRECTORY}' directory is not empty, data is already downloaded probably - aborting!")
        exit(1)

    # list of tuples to easily modify urls
    # format: (<run_num>, <min_seq_number>, <max_seq_number>)
    # eg. (8162, 80, 237)
    # will give results:
    #   https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/3/frame-r-008162-1-0080.fits.bz2
    #   https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/3/frame-r-008162-1-0081.fits.bz2
    #   https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/3/frame-r-008162-1-0082.fits.bz2
    #   and so on...
    urls_meta_numbers = [
        (8162, 80, 237),  # 80, 237
        # (8110, 11, 225)
    ]

    for (run_num, min_seq, max_seq) in urls_meta_numbers:
        for camera_col in range(1, 6):
            for seq_num in range(min_seq, max_seq):
                for band in ["g", "i", "r", "u", "z"]:
                    url = build_url(run_num, band, camera_col, seq_num, "fits.bz2")
                    download_or_log_at_fail(url, DATA_DIRECTORY)

                for img_type in ["irg", "thumb-irg"]:
                    url = build_url(run_num, img_type, camera_col, seq_num, "jpg")
                    download_or_log_at_fail(url, DATA_DIRECTORY)
