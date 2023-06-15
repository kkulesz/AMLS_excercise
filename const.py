import os

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ABSOLUTE_PATH, "data")
ALIGNED_DATA_DIR = os.path.join(DATA_DIR, "aligned")
COORDS_DATA_DIR = os.path.join(DATA_DIR, "coords")
TARGET_DATA_DIR = os.path.join(DATA_DIR, "targets")

IMG_ID_REGEX = "[0-9]{6}-[1-6]-[0-9]{4}"

PIECE_SHAPE = (256, 256)  # multiples of 64 if UnetV3
PIECE_ID_REGEX = f"{IMG_ID_REGEX}_[0-9]+\D"
PIECES_READY_DATA_DIR = os.path.join(DATA_DIR, "ready")
PIECES_READY_DATA_INPUTS_DIR = os.path.join(PIECES_READY_DATA_DIR, "inputs")
PIECES_READY_DATA_TARGETS_DIR = os.path.join(PIECES_READY_DATA_DIR, "targets")

INPUT_CHANNELS = 5
OUTPUT_CHANNELS = 3
LEARNING_RATE = 0.002
BATCH_SIZE = 20
ADAM_BETAS = (0.5, 0.999)
