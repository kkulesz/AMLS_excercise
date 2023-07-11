import os

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

IMAGES_TO_DOWNLOAD_METADATA = [
    # list of tuples to easily modify urls
    # format: (<run_num>, <min_seq_number>, <max_seq_number>)
    # eg. (8162, 80, 237)
    # will give results:
    #   https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/3/frame-r-008162-1-0080.fits.bz2
    #   https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/3/frame-r-008162-1-0081.fits.bz2
    #   https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/3/frame-r-008162-1-0082.fits.bz2
    #   and so on...
    (8162, 80, 80, 6, 6, False),  # 6th (8162, 80, 237); for the sake of comparison - 301/8162/6/frame-irg-008162-6-0080
    (8162, 81, 85, 6, 6, True),  # 6th (8162, 80, 237); my own selection, similar to test
    (8110, 11, 15, 1, 1, False),  # (8110, 11, 225); my own selection
    (3918, 213, 213, 3, 3, True),  # 3th (3918, 28, 434); used in repo
]

DATA_DIR = os.path.join(ABSOLUTE_PATH, "data")
ALIGNED_DATA_DIR = os.path.join(DATA_DIR, "aligned")
COORDS_DATA_DIR = os.path.join(DATA_DIR, "coords")
TARGET_DATA_DIR = os.path.join(DATA_DIR, "targets")

IMG_ID_REGEX = "[0-9]{6}-[1-6]-[0-9]{4}"

ORIGINAL_IMAGE_SHAPE = (1489, 2048)
PIECE_SHAPE = (32, 32)  # 128 is too much, 64 is fine
PIECE_ID_REGEX = f"{IMG_ID_REGEX}_[0-9]+\D"
PIECE_DIR_INPUT_NAME = "input_pieces"
PIECE_DIR_TARGET_NAME = "target_pieces"

VALIDATE_IMG = "008162-6-0080"
VALIDATE_DIR = os.path.join(DATA_DIR, "to_validate")
VALIDATE_INPUT_PATH = os.path.join(VALIDATE_DIR, f"{VALIDATE_IMG}.npy")
VALIDATE_TARGET_PATH = os.path.join(VALIDATE_DIR, f"{VALIDATE_IMG}_target.npy")

TEST_RATIO = 0.2
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.1

IMAGES_TO_ASSIGN_TO_TEST_SET = ["008162-6-0080"]

SPLITTED_DATA_DIR = os.path.join(ABSOLUTE_PATH, "..", "splitted")
TEST_DIR = os.path.join(SPLITTED_DATA_DIR, "test")
TRAIN_DIR = os.path.join(SPLITTED_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(SPLITTED_DATA_DIR, "validation")

INPUTS_DIR_NAME = "inputs"
TARGETS_DIR_NAME = "targets"

TEST_INPUTS_DIR = os.path.join(TEST_DIR, INPUTS_DIR_NAME)
TRAIN_INPUTS_DIR = os.path.join(TRAIN_DIR, INPUTS_DIR_NAME)
VALIDATION_INPUTS_DIR = os.path.join(VALIDATION_DIR, INPUTS_DIR_NAME)
TEST_TARGETS_DIR = os.path.join(TEST_DIR, TARGETS_DIR_NAME)
TRAIN_TARGETS_DIR = os.path.join(TRAIN_DIR, TARGETS_DIR_NAME)
VALIDATION_TARGETS_DIR = os.path.join(VALIDATION_DIR, TARGETS_DIR_NAME)

CSV_NAME = 'files.csv'
CSV_INPUT_COL = 'input'
CSV_TARGET_COL = 'target'

INPUT_CHANNELS = 5
OUTPUT_CHANNELS = 3
LEARNING_RATE = 0.0001  # 0.0001
BATCH_SIZE = 256  # 256
ADAM_BETAS = (0.5, 0.999)
START_EPOCH_FROM = 20
NUMBER_OF_EPOCHS = 60

VALIDATION_EPOCHS = 10

SAVE_MODEL_INTERVAL = 10
SAVE_INFERENCE_RESULT_IMG_INTERVAL = 5
VALIDATE_MODEL_INTERVAL = 1
TEST_MODEL_INTERVAL = 5

LOG_LOSS_ITERATION_INTERVAL = 50
VALIDATION_OUTPUT_DIR = os.path.join(ABSOLUTE_PATH, "validation_outputs")

THRESHOLD_VALUE = 0
