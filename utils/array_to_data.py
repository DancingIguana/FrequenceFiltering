import cv2
import numpy as np
def array_to_data(image: np.array):
    """
    Given an image, get the image in bytes for displaying 
    in PySimpleGUI

    Parameters:
    --------------------
    image: the matrix of the image

    Returns:
    --------------------
    The bytes of the image
    """
    imgbytes = cv2.imencode(".png", image)[1].tobytes()
    return imgbytes
