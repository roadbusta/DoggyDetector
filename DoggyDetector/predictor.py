# Import relevant libraries
import os
import numpy as np

from keras.applications import inception_v3

import cv2
import matplotlib.pyplot as plt


class Predictor():
    def predictor(image_path, model):

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

        # This is where I am up to, change the below to the array_to_tensor function


        #Convert image into tensor
        def array_to_tensor(input_array):
            list_of_tensors = [
                np.expand_dims(image, axis=0) for image in input_array
            ]
            return np.vstack(list_of_tensors)


        _single_test = array_to_tensor(_single_test).astype('float32') / 255

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
        dog_breed_predictions

        # Create a list of breeds
        DATADIR = "/Users/joe/Desktop/Temporary/Stanford Dogs Dataset/Images"
        CATEGORIES = os.listdir(DATADIR)
        CATEGORIES.remove(".DS_Store")

        print(CATEGORIES[dog_breed_predictions[0]])


# if __name__ == "__main__":
# model = model_from_pickle()
# image_path = "/Users/joe/Desktop/Screen Shot 2022-02-23 at 8.26.00 am.png"
