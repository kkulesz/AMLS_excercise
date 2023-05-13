import os
import bz2
import gzip

import const


def listdir_fullpath(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]


def decompress_bz2_files(list_of_files):
    bz2_files = filter(lambda file_name: file_name.endswith(".bz2"), files)
    for file in bz2_files:
        with bz2.open(file, "rb") as f:
            data = f.read()
        new_file_path = file[:-4]  # get rid of '.bz2' ending
        open(new_file_path, 'wb').write(data)
        print(f"Decompressed: {file}")


def decompress_gzip_files(list_of_files):
    gzip_files = filter(lambda file_name: file_name.endswith(".gz"), files)
    for file in gzip_files:
        with gzip.open(file, "rb") as f:
            data = f.read()
        new_file_path = file[:-3]  # get rid of '.gz' ending
        open(new_file_path, 'wb').write(data)
        print(f"Decompressed: {file}")


if __name__ == "__main__":
    files = listdir_fullpath(const.DATA_DIR)

    # decompress_bz2_files(files)
    decompress_gzip_files(files)
