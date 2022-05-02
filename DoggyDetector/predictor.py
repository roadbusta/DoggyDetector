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

        #Do this bottle neck thing
        inception_bottleneck = inception_v3.InceptionV3(weights='imagenet',
                                                        include_top=False,
                                                        pooling='avg')

        _single_test = inception_bottleneck.predict(_single_test,
                                                    batch_size=32,
                                                    verbose=0)

        # Create a list of breeds


        with open ('breed_list.pickle', 'rb') as fp:
            breeds = pickle.load(fp)

        #Perform prediction - This is list comprehension
        dog_breed_predictions = [
            np.argmax(model.predict(np.expand_dims(tensor, axis=0)))
            for tensor in _single_test
        ]

        #Print the indicies of the top 2 predictions
        a = model.predict(np.expand_dims(_single_test[0], axis=0))[0]
        ind = np.argpartition(a, -4)[-4:]

        print("Indices :", ind)

        #Sort the indices from biggest to smallest
        sorted_a = ind[np.argsort(a[ind])]
        print("Sorted array : ", sorted_a)

        # Find the top predictions
        for index in sorted_a:
            print(f"Index: {index} . Likelihood: {a[index]} . Breed: {breeds[index]} ")


        # print("Print model predictions")
        # # This appears to be the array that contains predictions
        # for tensor in _single_test:
        #     print(model.predict(np.expand_dims(tensor, axis=0)))

        # print("Print model predictions as a number")
        # for tensor in _single_test:
        #     np.argmax(model.predict(np.expand_dims(tensor, axis=0)))

        # print("Print length of predictions")
        # for tensor in _single_test:
        #     print(len(model.predict(np.expand_dims(tensor, axis=0))[0]))

        # print("Prediction certainty")
        # for tensor in _single_test:
        #     prediction_certainty = model.predict(np.expand_dims(tensor, axis=0))[0]
        #     prediction_index = np.argmax(
        #         model.predict(np.expand_dims(tensor, axis=0)))
        #     print(prediction_certainty[prediction_index])



        # dog_breed_predictions

        # Create a list of breeds


        with open ('breed_list.pickle', 'rb') as fp:
            breeds = pickle.load(fp)



        # return breeds[dog_breed_predictions[0]]

        return [(a[sorted_a[-1]], breeds[sorted_a[-1]]),
                (a[sorted_a[-2]],breeds[sorted_a[-2]])]  #Returns top two predictions


if __name__ == "__main__":
    model = model_from_pickle()
    path = "/Users/joe/Desktop/test.jpeg"

    predictor = Predictor()

    prediction = predictor.predict(image_path = path, model = model)
    print(prediction)
