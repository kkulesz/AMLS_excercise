import numpy as np


# source: https://medium.com/mlearning-ai/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f
def precision_score(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    precision = np.mean(intersect / total_pixel_pred)
    return precision


def recall_score(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    recall = np.mean(intersect / total_pixel_truth)
    return recall


def accuracy(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask == pred_mask)
    acc = np.mean(xor / (union + xor - intersect))
    return acc


def dice_coef(targs, pred):
    # intersect = np.sum(pred_mask * groundtruth_mask)
    # total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    # dice = np.mean(2 * intersect / total_sum)
    pred = (pred > 0).astype('float32')
    return 2. * (pred * targs).sum() / (pred + targs).sum()


def iou(groundtruth_mask, pred_mask):
    # intersect = np.sum(pred_mask * groundtruth_mask)
    # union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    # iou = np.mean(intersect / union)

    intersection = np.logical_and(groundtruth_mask, pred_mask)
    union = np.logical_or(groundtruth_mask, pred_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
