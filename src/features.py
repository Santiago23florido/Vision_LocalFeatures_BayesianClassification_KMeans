import cv2
import numpy as np


VALID_FEATURE_SPACES = ("rgb", "hsv", "ycrcb", "cbcr_grad")


def list_feature_spaces():
    return VALID_FEATURE_SPACES


def extract_feature_image(img_bgr, feature_space):
    if feature_space not in VALID_FEATURE_SPACES:
        raise ValueError(f"Unknown feature space '{feature_space}'.")

    if feature_space == "rgb":
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb.astype(np.float32), ("R", "G", "B")

    if feature_space == "hsv":
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return img_hsv.astype(np.float32), ("H", "S", "V")

    if feature_space == "ycrcb":
        img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        return img_ycrcb.astype(np.float32), ("Y", "Cr", "Cb")

    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(img_ycrcb)
    grad_x = cv2.Sobel(y_channel, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(y_channel, cv2.CV_32F, 0, 1, ksize=3)
    grad_norm = cv2.magnitude(grad_x, grad_y)
    feature_image = np.dstack(
        [
            cb_channel.astype(np.float32),
            cr_channel.astype(np.float32),
            grad_norm.astype(np.float32),
        ]
    )
    return feature_image, ("Cb", "Cr", "|grad(Y)|")


def flatten_feature_image(feature_image):
    if feature_image.ndim != 3:
        raise ValueError("feature_image must be an HxWxD array.")
    height, width, channels = feature_image.shape
    return feature_image.reshape(height * width, channels)
