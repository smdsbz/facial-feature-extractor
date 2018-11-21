# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image as PILImg
from matplotlib import image as mpimage

from matplotlib import pyplot as plt


def load_img(path):
    '''
    Args:
        path: (str) Path to image file

    Return:
        (np.ndarray) Image in tensor.
    '''
    img = mpimage.imread(path)
    img.setflags(write=True)
    return img
