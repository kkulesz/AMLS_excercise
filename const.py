import os

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ABSOLUTE_PATH, "data")
ALIGNED_DATA_DIR = os.path.join(DATA_DIR, "aligned")
COORDS_DATA_DIR = os.path.join(DATA_DIR, "coords")
TARGET_DATA_DIR = os.path.join(DATA_DIR, "targets")

IMG_ID_REGEX = "[0-9]{6}-[1-6]-[0-9]{4}"

ORIGINAL_IMAGE_SHAPE = (1489, 2048)
PIECE_SHAPE = (64, 64)  # multiples of 64 if UnetV3
PIECE_ID_REGEX = f"{IMG_ID_REGEX}_[0-9]+\D"
PIECES_READY_DATA_DIR = os.path.join(DATA_DIR, "ready")
PIECES_READY_DATA_INPUTS_DIR = os.path.join(PIECES_READY_DATA_DIR, "inputs")
PIECES_READY_DATA_TARGETS_DIR = os.path.join(PIECES_READY_DATA_DIR, "targets")

VALIDATE_IMG = "008162-6-0080"
VALIDATE_DIR = os.path.join(DATA_DIR, "to_validate")
VALIDATE_INPUT_PATH = os.path.join(VALIDATE_DIR, f"{VALIDATE_IMG}.npy")
VALIDATE_TARGET_PATH = os.path.join(VALIDATE_DIR, f"{VALIDATE_IMG}_target.npy")

INPUT_CHANNELS = 5
OUTPUT_CHANNELS = 3
LEARNING_RATE = 0.002
BATCH_SIZE = 100
ADAM_BETAS = (0.5, 0.999)
NUMBER_OF_EPOCHS = 5
BILINEAR = True

SAVE_MODEL_INTERVAL = 5

THRESHOLD_VALUE = 0
