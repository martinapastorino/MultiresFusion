# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:37:47 2022

@author: marti
"""

import numpy as np
import random
from scipy import ndimage
from skimage.morphology import erosion, disk

# ER color palette
palette = {0 : (144, 7, 48),   # built-up 
           1 : (241, 238, 79),     # crop soil 
           2 : (21, 75, 29), # trees 
           3 : (180, 255, 146), # grass 
           4 : (251, 154, 153), # bare soil
           5 : (31, 35, 180), # water 
           6 : (0, 0, 0)} # unlabeled 

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

