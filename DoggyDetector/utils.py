#Import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

def cat_list(DATADIR = "../raw_data/Images"):
    categories = os.listdir(DATADIR)
    categories.remove(".DS_Store")
    return cat_list
