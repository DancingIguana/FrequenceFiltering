import cv2
import os
import numpy as np

def load_image_gray(filename: str) -> np.array:
    """
    Given a filename, load the image in grayscale with cv2 and make sure
    that its width and height are even.

    Parameters:
    -------------------
    filename: the file path of the image

    Returns:
    -------------------
    the grayscale image matrix

    """
    if os.path.exists(filename):
        cv2_image = cv2.imread(filename)

        # Have even number of rows and columns
        height, width = cv2_image.shape[:2]
        if height % 2 == 1: height += 1
        if width % 2 == 1: width += 1

        cv2_image = cv2.resize(cv2_image,(width, height))

        gray_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        return gray_image

def load_image_rgb(filename):
    if os.path.exists(filename):
        cv2_image = cv2.imread(filename)

        # Have even number of rows and columns
        height, width = cv2_image.shape[:2]
        if height % 2 == 1: height += 1
        if width % 2 == 1: width += 1

        cv2_image = cv2.resize(cv2_image,(width, height))

        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        RGB_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        return RGB_image