#Import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import random
import pickle
from google.cloud import storage
import joblib
from termcolor import colored

"""
Functions related primairy to the loading and saving of data
"""
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


def data_to_pickle(X, y, pickle_path="./data/Pickle Files/"):
    """
    Converts the data into pickle files for easier loading
    in the future
    """

    pickle_out = open(pickle_path + "X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(pickle_path + "y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def data_from_pickle(pickle_path="./data/Pickle Files/"):
    """
    Takes data from the pickle files and loads them as X and y values
    """

    #Load the pickle files
    pickle_in = open(pickle_path + "X.pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open(pickle_path + "y.pickle", "rb")
    y = pickle.load(pickle_in)

    return X, y


#Data to GCP
def file_to_gcp(SOURCE_FILE_NAME, BUCKET_NAME,BUCKET_DESTINATION, rm = False):
    """
    Sends the file to google cloud platform.
    Set rm = True if the local file is to be deleted as well
    """

    client = storage.Client().bucket(BUCKET_NAME)

    blob = client.blob(BUCKET_DESTINATION)

    blob.upload_from_filename(SOURCE_FILE_NAME)

    print(colored(f"=> {SOURCE_FILE_NAME} uploaded to bucket {BUCKET_NAME} inside {BUCKET_DESTINATION}",
                  "green"))
    if rm:
        os.remove(SOURCE_FILE_NAME)


#Data from GCP (WORK IN PROGRESS)
def file_from_gcp(BUCKET_NAME, BUCKET_PICKLE_LOCATION,DESTINATION_FILE_NAME):

    """
    Takes the file from gcp and saves it to the stated destination
    """

    storage_client = storage.Client()

    bucket = storage_client.bucket(BUCKET_NAME)
    # Construct a client side representation of a blob.
    blob = bucket.blob(BUCKET_PICKLE_LOCATION)
    blob.download_to_filename(DESTINATION_FILE_NAME)
    print(colored("Downloaded storage object {} from bucket {} to local file {}.".format(
            BUCKET_PICKLE_LOCATION, BUCKET_NAME, DESTINATION_FILE_NAME), "green"))


#Model to pickle
def model_to_pickle(model, pickle_path="./data/Pickle Files/"):
    """
    Converts the model into pickle files for easier loading
    in the future
    """
    pickle_out = open(pickle_path + "model.pickle", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()


#Model from pickle
def model_from_pickle(pickle_path="./data/Pickle Files/"):
    """
    Takes model from the pickle files and loads it
    """

    #Load the pickle files
    pickle_in = open(pickle_path + "model.pickle", "rb")
    model = pickle.load(pickle_in)

    return model


### Not sure if a code needs to be developed for this or not

# Save files (?) Not sure if this is always required
# np.save('bottleneck_features_train_inception.npy', train_i_bf)
# np.save('bottleneck_features_val_inception.npy', val_i_bf)
# np.save('bottleneck_features_test_inception.npy', test_i_bf)

# # load the bottleneck features saved earlier
# train_data = np.load('bottleneck_features_train_inception.npy')
# val_data = np.load('bottleneck_features_val_inception.npy')
# test_data = np.load('bottleneck_features_test_inception.npy')

#Save the weights
# model.save_weights('inception_model_2.h5')


# if __name__ == "__main__":

#     BUCKET_NAME = "doggy-detector-2022-bucket"
#     BUCKET_PICKLE_LOCATION = "Pickle Files/y.pickle"
#     DESTINATION_FILE_NAME = "./data/Pickle Files/y.pickle"
#     file_from_gcp(BUCKET_NAME, BUCKET_PICKLE_LOCATION, DESTINATION_FILE_NAME)
