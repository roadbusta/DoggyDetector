#Import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

def category_list(DATADIR= "../raw_data/Images"):
    """
    Creates a category list based on the folders in [DATADIR]
    """
    categories = os.listdir(DATADIR)
    categories.remove(".DS_Store")
    return categories

def breed_list(DATADIR = "../raw_data/Images"):
    """
    Cleans up the category list based on the folders in [DATADIR]
    """
    breeds = []
    categories = os.listdir(DATADIR)
    categories.remove(".DS_Store")
    for category in categories:
        breed = category[10:]
        breed = breed.replace("_", " " )
        breed = breed.title()
        breeds.append(breed)

    return breeds


def create_training_data(CATEGORIES, IMG_SIZE = 224, DATADIR ="../raw_data/Images" ):
    '''
    Creates a training data

    DATADIR: The directory containing the raw data
    CATEGORIES: taken as the index of the names of the folders.
    IMG_SIZE: Outputted image size. (IMG_SIZE by IMG_SIZE)

    Returns X and y as a shuffled list of the data and labels.
    NOTE: y has not been one hot encoded.
    '''
    X = []
    y = []
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(
            DATADIR,
            category)  #This is the directory for either "Dog" or "Cat"
        class_num = CATEGORIES.index(
            category)  #This is the index for categories
        for img in os.listdir(path):  #For each image in the path
            try:
                img_array = cv2.imread(os.path.join(
                    path, img))  #The second argument returns a grayscale image
                new_array = cv2.resize(
                    img_array, (IMG_SIZE, IMG_SIZE))  # Resize the image
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    #Shuffle the data
    random.shuffle(training_data)

    #Assign to X and y
    for features, label in training_data:
        X.append(features)
        y.append(label)

    return X, y






# if __name__ == "__main__":
#     print(breed_list())
