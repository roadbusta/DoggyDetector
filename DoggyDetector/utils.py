import numpy as np
from DoggyDetector.data import breed_list, model_from_pickle




 # Create a function that converts arrays into tensors
def array_to_tensor(input_array):
    """
    A function to create arrays into tensors for Keras
    """
    list_of_tensors = [np.expand_dims(image, axis=0) for image in input_array]
    return np.vstack(list_of_tensors)

# Predict breed
def predict_breed():
    breeds = breed_list()
    model = model_from_pickle()
