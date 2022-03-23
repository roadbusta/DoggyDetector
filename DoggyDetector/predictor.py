# Import relevant libraries
from email.mime import image
from DoggyDetector.data import breed_list, model_from_pickle
from DoggyDetector.utils import array_to_tensor

from tensorflow import keras
import os
import numpy as np
from keras.applications import inception_v3
import cv2 #Commenting this out to troubleshoot docker
import matplotlib.pyplot as plt
import pickle


class Predictor():
    def predict(self,image_path, model):

        #Load the image
        single_test = cv2.imread(image_path)
        print("Image loaded \n")

        # Convert into a list
        _single_test = [single_test]


        # Resize the image
        IMG_SIZE = 224
        temp_list = []
        for image in _single_test:
            temp_list.append(cv2.resize(image, (IMG_SIZE, IMG_SIZE)))
        _single_test = temp_list

        # Convert to an array
        _single_test = np.array(_single_test)

        # Convert the array into a tensor
        _single_test = array_to_tensor(_single_test).astype('float32') / 255
        print("Converted into tensor \n")
        #Do this bottle neck thing
        inception_bottleneck = inception_v3.InceptionV3(weights='imagenet',
                                                        include_top=False,
                                                        pooling='avg')

        _single_test = inception_bottleneck.predict(_single_test,
                                                    batch_size=32,
                                                    verbose=0)

        #Perform prediction
        dog_breed_predictions = [
            np.argmax(model.predict(np.expand_dims(tensor, axis=0)))
            for tensor in _single_test
        ]
        print("Prediction made \n")
        # dog_breed_predictions

        # Create a list of breeds
        print("Current working directory according to predictor.py")
        print(os.getcwd())

        print("Files in current directory according to fast.py")
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for f in files:
            print(f)

        with open ('breed_list.pickle', 'rb') as fp:
            breeds = pickle.load(fp)

        print("Breed list loaded \n")

        return breeds[dog_breed_predictions[0]]


# if __name__ == "__main__":
#     model = model_from_pickle()
#     path = "/Users/joe/Desktop/BREED Hero Desktop_0113_french_bulldog.webp"

#     predictor = Predictor()

#     predictor.predict(image_path = path, model = model)
