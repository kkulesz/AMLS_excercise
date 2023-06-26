import const
import utils

from data_acquistion_and_alignment.download_data import download_indicated_images
from data_acquistion_and_alignment.decompress_data import decompress_everything_needed
from data_acquistion_and_alignment.align_data import read_fits_files_and_align_them
from data_acquistion_and_alignment.process_coords import process_everything_needed
from data_preparation.split_data import split_data_based_on_ratio
from data_acquistion_and_alignment.materialize_target import materialize_target_for_directory
from data_preparation.split_into_smaller_pieces import split_images_in_directory


def _prepare_dirs(list_of_dirs):
    for dir in list_of_dirs:
        utils.create_dir_if_doesnt_exist(dir)
        # print(f"Created {dir}...")


def main():
    metadata_of_images_to_download = const.IMAGES_TO_DOWNLOAD_METADATA

    data_dir = const.DATA_DIR
    aligned_data_dir = const.ALIGNED_DATA_DIR
    coords_dir = const.COORDS_DATA_DIR
    splitted_data_dir = const.SPLITTED_DATA_DIR

    test_dir = const.TEST_DIR
    train_dir = const.TRAIN_DIR
    validation_dir = const.VALIDATION_DIR

    test_inputs_dir = const.TEST_INPUTS_DIR
    train_inputs_dir = const.TRAIN_INPUTS_DIR
    validation_inputs_dir = const.VALIDATION_INPUTS_DIR

    test_targets_dir = const.TEST_TARGETS_DIR
    train_targets_dir = const.TRAIN_TARGETS_DIR
    validation_targets_dir = const.VALIDATION_TARGETS_DIR

    all_dirs = [
        data_dir, aligned_data_dir, coords_dir, splitted_data_dir,
        test_dir, train_dir, validation_dir,
        test_inputs_dir, train_inputs_dir, validation_inputs_dir,
        test_targets_dir, train_targets_dir, validation_targets_dir
    ]

    _prepare_dirs(all_dirs)
    print("\nStarting data pipline...")

    # download_indicated_images(data_dir, metadata_of_images_to_download)
    # print("Data downloaded...")
    # decompress_everything_needed(data_dir)
    # print("Data decompressed...")
    # read_fits_files_and_align_them(data_dir, aligned_data_dir)
    # print("Data aligned...")
    # process_everything_needed(coords_dir, data_dir)
    # print("Cords processed...")

    split_data_based_on_ratio(
        aligned_data_dir, coords_dir,
        test_inputs_dir, train_inputs_dir, validation_inputs_dir
    )
    print("Data splitted...")

    print("Materializing targets...")
    materialize_target_for_directory(test_inputs_dir, test_targets_dir, coords_dir)
    print("\tTest done...")
    materialize_target_for_directory(train_inputs_dir, train_targets_dir, coords_dir)
    print("\tTrain done...")
    materialize_target_for_directory(validation_inputs_dir, validation_targets_dir, coords_dir)
    print("\tValidation done...")
    print("Materializing targets done...")

    print(f"Splitting images into pieces...")
    split_images_in_directory(test_dir, test_inputs_dir, test_targets_dir)
    print("\tTest done...")
    split_images_in_directory(train_dir, train_inputs_dir, train_targets_dir)
    print("\tTrain done...")
    split_images_in_directory(validation_dir, validation_inputs_dir, validation_targets_dir)
    print("\tValidation done...")
    print("Splitting images into pieces done...")

    print("\nData pipline finished.")


if __name__ == "__main__":
    main()
