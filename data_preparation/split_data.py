import re
import pandas as pd
import shutil

import utils
import const


def get_images_with_gal_and_star_count(images_dir, coords_dir):
    coords_files = utils.listdir_fullpath(coords_dir)
    gals_files = list(filter(lambda f: re.search("gal", f), coords_files))
    stars_files = list(filter(lambda f: re.search("star", f), coords_files))

    image_files = utils.listdir_fullpath(images_dir)

    img_gal_star_files_tuples = []
    for img_f in image_files:
        img_id = re.search(const.IMG_ID_REGEX, img_f).group()
        gal_f = next(g for g in gals_files if img_id in g)
        star_f = next(s for s in stars_files if img_id in s)

        img_gal_star_files_tuples.append(
            (img_f, gal_f, star_f)
        )

    img_with_gal_and_star_count = []
    for (img_f, gal_f, star_f) in img_gal_star_files_tuples:
        gal_count = pd.read_csv(gal_f).size
        star_count = pd.read_csv(star_f).size
        img_with_gal_and_star_count.append(
            (img_f, gal_count, star_count)
        )

    return img_with_gal_and_star_count


def should_sample_go_into_this_split(set_size, max_set_size, ratio_overall, ratio_so_far, ratio):
    return \
            (set_size < max_set_size) and \
            ((ratio_overall > ratio_so_far and ratio_so_far < ratio) or
             (ratio_overall < ratio_so_far and ratio_so_far > ratio))


def add_sample_to_set(dset, dstats, img_file, gal_count, stars_count):
    dset.append(img_file)
    dstats = (dstats[0] + gal_count, dstats[1] + stars_count)
    return dset, dstats


def calc_ratio(stats):
    gals, stars = stats
    return gals / stars if stars > 0 else 0


def split_data(images_with_counts, images_to_assign_to_test):
    num_images = len(images_with_counts)
    num_train = int(num_images * const.TRAIN_RATIO)
    num_test = int(num_images * const.TEST_RATIO)
    num_validation = num_images - num_train - num_test

    gals_overall = sum(list(map(lambda x: x[1], images_with_counts)))
    stars_overall = sum(list(map(lambda x: x[2], images_with_counts)))

    ratio_overall = gals_overall / stars_overall

    train_set = []
    test_set = []
    validation_set = []

    train_stats = (0, 0)
    test_stats = (0, 0)
    validation_stats = (0, 0)

    # iterate through data and assign images to test beforehand
    images_with_counts_filtered = []
    for (img_file, gal_count, stars_count) in images_with_counts:
        img_id = re.search(const.IMG_ID_REGEX, img_file).group()
        if img_id in images_to_assign_to_test:
            test_set, test_stats = add_sample_to_set(test_set, test_stats, img_file, gal_count, stars_count)
        else:
            images_with_counts_filtered.append((img_file, gal_count, stars_count))

    # iterate through data that was not assigned beforehand
    for (img_file, gal_count, stars_count) in images_with_counts_filtered:
        ratio = gal_count / stars_count

        train_ratio_so_far = calc_ratio(train_stats)
        test_ratio_so_far = calc_ratio(test_stats)
        validation_ratio_so_far = calc_ratio(validation_stats)

        if should_sample_go_into_this_split(len(train_set), num_train, ratio_overall, train_ratio_so_far, ratio):
            train_set, train_stats = add_sample_to_set(train_set, train_stats, img_file, gal_count, stars_count)
        elif should_sample_go_into_this_split(len(test_set), num_test, ratio_overall, test_ratio_so_far, ratio):
            test_set, test_stats = add_sample_to_set(test_set, test_stats, img_file, gal_count, stars_count)
        elif should_sample_go_into_this_split(len(validation_set), num_validation, ratio_overall, validation_ratio_so_far, ratio):
            validation_set, validation_stats = add_sample_to_set(validation_set, validation_stats, img_file, gal_count, stars_count)
        # no good place to assign sample -> assign it to wherever there is a place for it
        elif len(train_set) < num_train:
            train_set, train_stats = add_sample_to_set(train_set, train_stats, img_file, gal_count, stars_count)
        elif len(test_set) < num_test:
            test_set, test_stats = add_sample_to_set(test_set, test_stats, img_file, gal_count, stars_count)
        elif len(validation_set) < num_validation:
            validation_set, validation_stats = add_sample_to_set(validation_set, validation_stats, img_file, gal_count, stars_count)

    print(f"{ratio_overall} - ratio overall")
    print(f"{calc_ratio(train_stats)} - train ratio")
    print(f"{calc_ratio(test_stats)} - test ratio")
    print(f"{calc_ratio(validation_stats)} - validation ratio")
    return validation_set, train_set, test_set


def copy_files_into_destination_dir(files, dest_dir):
    for file in files:
        shutil.copy(file, dest_dir)


def _prepare_dirs():
    # creating '/splitted'
    utils.create_dir_if_doesnt_exist(const.SPLITTED_DATA_DIR)
    # creating '/splitted/[test/train/validation]'
    utils.create_dir_if_doesnt_exist(const.TEST_DIR)
    utils.create_dir_if_doesnt_exist(const.TRAIN_DIR)
    utils.create_dir_if_doesnt_exist(const.VALIDATION_DIR)
    # creating '/splitted/[test/train/validation]/inputs'
    utils.create_dir_if_doesnt_exist(const.TEST_INPUTS_DIR)
    utils.create_dir_if_doesnt_exist(const.TRAIN_INPUTS_DIR)
    utils.create_dir_if_doesnt_exist(const.VALIDATION_INPUTS_DIR)


if __name__ == "__main__":
    _prepare_dirs()

    images_with_counts = get_images_with_gal_and_star_count(
        const.ALIGNED_DATA_DIR, const.COORDS_DATA_DIR
    )

    validation_set, train_set, test_set = split_data(images_with_counts, const.IMAGES_TO_ASSIGN_TO_TEST_SET)

    copy_files_into_destination_dir(test_set, const.TEST_INPUTS_DIR)
    copy_files_into_destination_dir(train_set, const.TRAIN_INPUTS_DIR)
    copy_files_into_destination_dir(validation_set, const.VALIDATION_INPUTS_DIR)
