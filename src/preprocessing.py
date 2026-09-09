import cv2
import numpy as np


def gaussian_blur(
    frame,
    kernel_size=(5, 5)
):

    return cv2.GaussianBlur(
        frame,
        kernel_size,
        0
    )


def grayscale(frame):

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


def hsv(frame):

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


def canny(
    frame,
    low_threshold=50,
    high_threshold=150
):

    gray = grayscale(frame)

    blurred = gaussian_blur(gray)

    return cv2.Canny(
        blurred,
        low_threshold,
        high_threshold
    )


def white_yellow_mask(frame):

    hsv_image = hsv(frame)

    white = cv2.inRange(
        hsv_image,
        np.array([0, 0, 150]),
        np.array([180, 90, 255])
    )

    yellow = cv2.inRange(
        hsv_image,
        np.array([12, 55, 90]),
        np.array([45, 255, 255])
    )

    return cv2.bitwise_or(
        white,
        yellow
    )