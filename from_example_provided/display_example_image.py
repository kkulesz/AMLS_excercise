import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

if __name__ == "__main__":
    with open("003918-3-0213_img.pkl", "rb") as f:
        nn_img = pkl.load(f)
    # print(nn_img.shape)
    img = nn_img[:, :, :3]  # take irg channels for plotting
    img = np.maximum(0, img)
    img = np.power(img, 0.5)  # square root to make the high value pixels less dominant
    plt.figure()
    plt.axis("off")
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    plt.imshow(img)
    # plt.show()
    # plt.savefig("003918-3-0213_img.pdf")
