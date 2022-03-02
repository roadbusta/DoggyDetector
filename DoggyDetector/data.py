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
def pickle_to_gcp(local_pickle_name, BUCKET_NAME, rm = False):
    """
    Sends the pickle file to google cloud platform.
    Set rm = True if the local file is to be deleted as well
    """

    client = storage.Client().bucket(BUCKET_NAME)

    local_pickle_file = f"./data/Pickle Files/{local_pickle_name}"

    storage_location = f"Pickle Files/{local_pickle_name}"

    blob = client.blob(storage_location)

    blob.upload_from_filename(local_pickle_file)

    print(colored(f"=> {local_pickle_name} uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                  "green"))
    if rm:
        os.remove(local_pickle_file)


#Data from GCP (WORK IN PROGRESS)
def pickle_from_gcp(local_pickle_name, BUCKET_NAME, rm=False):

    # """
    # Takes data from the pickle files and loads them as X and y values
    # """

    # #Load the pickle files
    # pickle_in = open(pickle_path + "X.pickle", "rb")
    # X = pickle.load(pickle_in)

    # pickle_in = open(pickle_path + "y.pickle", "rb")
    # y = pickle.load(pickle_in)

    # return X, y

    # """
    # Sends the pickle file to google cloud platform.
    # Set rm = True if the local file is to be deleted as well
    # """

    # client = storage.Client().bucket(BUCKET_NAME)

    # local_pickle_file = f"./data/Pickle Files/{local_pickle_name}"

    # storage_location = f"Pickle Files/{local_pickle_name}"

    # blob = client.blob(storage_location)

    # blob.upload_from_filename(local_pickle_file)

    # print(
    #     colored(
    #         f"=> {local_pickle_name} uploaded to bucket {BUCKET_NAME} inside {storage_location}",
    #         "green"))
    # if rm:
    #     os.remove(local_pickle_file)

    return None







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


# Upload model to GCP
# def upload_model_to_gcp(model, BUCKET_NAME, STORAGE_LOCATION):
#     """
#     Dumps model into a .joblib file then uploads to google cloud
#     """
#     # joblib.dump(model, 'model.joblib')
#     # print("saved model.joblib locally")

#     # client = storage.Client()

# bucket = client.bucket(BUCKET_NAME)

# blob = bucket.blob(STORAGE_LOCATION)

# blob.upload_from_filename('model.joblib')

# print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

#This needs to be altered with this code:
# def storage_upload(rm=False):
# client = storage.Client().bucket(BUCKET_NAME)

# local_model_name = 'model.joblib'
# storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
# blob = client.blob(storage_location)
# blob.upload_from_filename('model.joblib')
# print(colored(f"=> model.joblib uploaded to bucket {BUCKET_NAME} inside {storage_location}",
#               "green"))
# if rm:
#     os.remove('model.joblib')

# Download model from GCP




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
#     pickle_X = "X.pickle"
#     pickle_y = "y.pickle"
#     BUCKET_NAME = "doggy-detector-2022-bucket"

#     pickle_to_gcp(pickle_y, BUCKET_NAME)
#     pickle_to_gcp(pickle_X, BUCKET_NAME)
