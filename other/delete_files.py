import os
import re

import const
import utils


def delete_gz(dir):
    files = utils.listdir_fullpath(dir)
    gz_files = list(filter(lambda f: f.endswith("gz"), files))
    for f in gz_files:
        os.remove(f)


def delete_bz2(dir):
    files = utils.listdir_fullpath(dir)
    gz_files = list(filter(lambda f: f.endswith("bz2"), files))
    for f in gz_files:
        os.remove(f)


if __name__ == "__main__":
    delete_gz(const.DATA_DIR)
    delete_bz2(const.DATA_DIR)
