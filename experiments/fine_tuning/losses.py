from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np
import cv2


def rectified_mae_loss(true, test):
    assert true.shape[0] == test.shape[0], \
        "True and Test batches must have the same shape."
    assert true.dtype == test.dtype and (true.dtype == np.uint8 or true.dtype == np.float32), \
        "That's a wrong dtype" + str(true.dtype) + " " + str(test.dtype)

    pixel_range = 255 if test.dtype == np.uint8 else 1
    area = true.shape[1] * true.shape[2] * true.shape[3]

    losses = [np.sum(cv2.absdiff(true[i] / pixel_range, test[i] / pixel_range)) / area for i in range(true.shape[0])]

    return sum(losses) / len(losses)


# -1 to 1, 1 means the images are identical
def ssim_loss(true, test):
    assert true.shape[0] == test.shape[0], "True and Test batches must have the same shape."
    losses = [ssim(true[i], test[i], multichannel=True) for i in range(true.shape[0])]
    return sum(losses) / true.shape[0]


def psnr_loss(true, test):
    assert true.shape[0] == test.shape[0], "True and Test batches must have the same shape."
    losses = [psnr(true, test) for i in range(true.shape[0])]
    return sum(losses) / true.shape[0]

def combined_loss(true, test):
    return mse(true, test) + (1 - ssim_loss(true, test)) + psnr_loss(true, test)

