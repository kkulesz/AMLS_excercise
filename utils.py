import os


def listdir_fullpath(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]
