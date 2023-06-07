import os

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ABSOLUTE_PATH, "data")
ALIGNED_DATA_DIR = os.path.join(DATA_DIR, "aligned")
COORDS_DATA_DIR = os.path.join(DATA_DIR, "coords")
TARGET_DATA_DIR = os.path.join(DATA_DIR, "targets")

IMG_ID_REGEX = "[0-9]{6}-[1-6]-[0-9]{4}"
