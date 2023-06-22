import os

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ABSOLUTE_PATH, "data")
ALIGNED_DATA_DIR = os.path.join(DATA_DIR, "aligned")
COORDS_DATA_DIR = os.path.join(DATA_DIR, "coords")
TARGET_DATA_DIR = os.path.join(DATA_DIR, "targets")

IMG_ID_REGEX = "[0-9]{6}-[1-6]-[0-9]{4}"

ORIGINAL_IMAGE_SHAPE = (1489, 2048)
PIECE_SHAPE = (64, 64)  # 128 is too much
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
SPLITTED_DATA_DIR = os.path.join(DATA_DIR, "splitted")
TEST_DIR = os.path.join(SPLITTED_DATA_DIR, "test")
TRAIN_DIR = os.path.join(SPLITTED_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(SPLITTED_DATA_DIR, "validation")

INPUT_CHANNELS = 5
OUTPUT_CHANNELS = 3
LEARNING_RATE = 0.02
BATCH_SIZE = 100
ADAM_BETAS = (0.5, 0.999)
NUMBER_OF_EPOCHS = 20
BILINEAR = True

SAVE_MODEL_INTERVAL = 5

THRESHOLD_VALUE = 0
