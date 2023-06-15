import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def listdir_fullpath(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]


def create_dir_if_doesnt_exist(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def display_image(img):
    img = img[:, :, :3]  # take irg channels for plotting
    img = np.maximum(0, img)
    img = np.power(img, 0.5)  # square root to make the high value pixels less dominant
    plt.figure()
    plt.axis("off")
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    plt.imshow(img)
    plt.show()


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Working on {device} device.")

    return device

