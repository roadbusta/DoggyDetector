# Import relevant custom utilities
from DoggyDetector.data import save_model_locally, storage_upload, category_list, create_training_data, data_from_pickle, model_to_pickle, data_to_pickle, file_from_gcp, file_to_gcp, pickle_from_gcp
from DoggyDetector.model import init_model
from DoggyDetector.utils import array_to_tensor

# Import relevant libraries
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.applications import inception_v3
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import applications
import pickle
import os
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient



MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[AUS] [MEL] [roadbusta] inception + v1"

class Trainer():


    def train_local_data(self, n = None,pickle_source = True, make_file = True):
        """
        Trains a model locally
        n is the number of images you want to create the model based on
        set pickle to True if the data is already in a pickle file
        """
        #Load the data for pickle if it exists, otherwise from the raw data
        if pickle_source:
            X, y = data_from_pickle(make_file= make_file)
        else:
            categories = category_list(make_file= make_file)
            X, y = create_training_data(categories, make_file= make_file)
            data_to_pickle(X, y)

        #Create a smaller sample size
        X_small = X[:n]
        y_small = y[:n]

        # Transform X and y as required
        X_in = np.array(X_small)
        num_classes = len(set(y_small))
        # Convert y into categorical data
        y_in = to_categorical(y_small, num_classes)

        # Create the train, test and validation sets
        first_split = int(len(X_small) /6.)
        second_split = first_split + int(len(X_small) * 0.2)
        X_test, X_val, X_train = X_in[:first_split], X_in[first_split:second_split], X_in[second_split:]
        y_test, y_val, y_train = y_in[:first_split], y_in[first_split:second_split], y_in[second_split:]

        # pre-process the data for Keras - Converts to (224, 224) and converts into a numpy array using PIL.
        train_tensors = array_to_tensor(X_train).astype('float32')/255
        val_tensors = array_to_tensor(X_val).astype('float32')/255
        test_tensors = array_to_tensor(X_test).astype('float32')/255

        # Create predict files (?) Note: I'm not exactly sure what this step does exactly
        input_size=224 # This is the hard-coded image size. Change this if the images are resized
        num_classes=120 # This is the hard-coded number of classes. Change this if the number of classes change
        S=1
        train_len=len(X_train)

        #Do the bottleneck thing
        inception_bottleneck=inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        train_i_bf = inception_bottleneck.predict(train_tensors, batch_size=32, verbose=0)
        val_i_bf = inception_bottleneck.predict(val_tensors, batch_size=32, verbose=0)
        test_i_bf = inception_bottleneck.predict(test_tensors, batch_size=32, verbose=0)

        batch_size=32
        epochs=50

        # Initialise the model
        model = init_model(num_classes = num_classes)

        # Fit the model
        model.fit(train_i_bf, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(val_i_bf, y_val),
                verbose = 0)

        #Evaluate the model- not sure if this is needed
        (eval_loss, eval_accuracy) = model.evaluate(val_i_bf,
                                                    y_val,
                                                    batch_size=batch_size,
                                                    verbose=0)



        MODEL_NAME = "Inception"
        MODEL_VERSION = "V1"

        self.experiment_name = "[AUS] [MEL] [roadbusta] inception + v1"


        metric_name = "accuracy"
        metric_value = eval_accuracy
        # self.mlflow_log_param(param_name, param_value)
        self.mlflow_log_metric(metric_name, metric_value)


        #Save the model as a pickle file
        model_to_pickle(model, make_file= make_file)

        #Save model locally as model.joblib
        save_model_locally(model)

        print("Model Trained")


    def train_GCP_data(self, n=None, pickle_source=True, make_file = True):
        #Apply makefile trigger to absolute working directory
        awd = ".."
        if make_file:
            awd = "."


        #Load the data from Pickle File in GCP

        if pickle_source:
            BUCKET_NAME = "doggy-detector-2022-bucket-v2"

            #Load y pickle
            BUCKET_PICKLE_LOCATION = "Pickle Files/y.pickle"

            y_pickle_in = pickle_from_gcp(BUCKET_NAME, BUCKET_PICKLE_LOCATION)
            y = pickle.loads(y_pickle_in)
            #Load x pickle
            BUCKET_PICKLE_LOCATION = "Pickle Files/X.pickle"
            X_pickle_in = pickle_from_gcp(BUCKET_NAME, BUCKET_PICKLE_LOCATION)
            X = pickle.loads(X_pickle_in)


            print("Data has been loaded from pickle")

        else:
            ##Note this hasn't been tested###
            #Download the image files and train from scratch
            BUCKET_NAME = "doggy-detector-2022-bucket-v2"

            #Load y pickle
            BUCKET_PICKLE_LOCATION = "data/"
            DESTINATION_FILE_NAME = awd + "/raw_data/"
            file_from_gcp(BUCKET_NAME, BUCKET_PICKLE_LOCATION,
                          DESTINATION_FILE_NAME)

            categories = category_list(make_file = make_file)
            X, y = create_training_data(categories, make_file= make_file)
            #Save X and y as pickle files locally
            data_to_pickle(X, y)
            ### ###


        #Create a smaller sample size
        X_small = X[:n]
        y_small = y[:n]

        # Transform X and y as required
        X_in = np.array(X_small)
        num_classes = len(set(y_small))
        # Convert y into categorical data
        y_in = to_categorical(y_small, num_classes)

        # Create the train, test and validation sets
        first_split = int(len(X_small) / 6.)
        second_split = first_split + int(len(X_small) * 0.2)
        X_test, X_val, X_train = X_in[:first_split], X_in[
            first_split:second_split], X_in[second_split:]
        y_test, y_val, y_train = y_in[:first_split], y_in[
            first_split:second_split], y_in[second_split:]

        # pre-process the data for Keras - Converts to (224, 224) and converts into a numpy array using PIL.
        train_tensors = array_to_tensor(X_train).astype('float32') / 255
        val_tensors = array_to_tensor(X_val).astype('float32') / 255
        test_tensors = array_to_tensor(X_test).astype('float32') / 255

        # Create predict files (?) Note: I'm not exactly sure what this step does exactly
        input_size = 224  # This is the hard-coded image size. Change this if the images are resized
        num_classes = 120  # This is the hard-coded number of classes. Change this if the number of classes change
        S = 1
        train_len = len(X_train)

        #Do the bottleneck thing
        inception_bottleneck = inception_v3.InceptionV3(weights='imagenet',
                                                        include_top=False,
                                                        pooling='avg')
        train_i_bf = inception_bottleneck.predict(train_tensors,
                                                  batch_size=32,
                                                  verbose=0)
        val_i_bf = inception_bottleneck.predict(val_tensors,
                                                batch_size=32,
                                                verbose=0)
        test_i_bf = inception_bottleneck.predict(test_tensors,
                                                 batch_size=32,
                                                 verbose=0)

        batch_size = 32
        epochs = 50

        # Initialise the model
        model = init_model(num_classes=num_classes)

        # Fit the model
        model.fit(train_i_bf,
                  y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(val_i_bf, y_val),
                  verbose = 1)

        # Evaluate the model- not sure if this is needed
        (eval_loss, eval_accuracy) = model.evaluate(val_i_bf,
                                                    y_val,
                                                    batch_size=batch_size,
                                                    verbose=0)

        #Log on ML Flow
        MODEL_NAME = "Inception"
        MODEL_VERSION = "V1"

        self.experiment_name = "[AUS] [MEL] [roadbusta] inception + v1"

        metric_name = "accuracy"
        metric_value = eval_accuracy
        # self.mlflow_log_param(param_name, param_value)
        self.mlflow_log_metric(metric_name, metric_value)

        print(f"Results with accuracy= {metric_value} has been uploaded ML Flow")


        #Save the model as a model.joblib file
        save_model_locally(model)

        # Upload the modle to Google Cloud Platform

        MODEL_NAME = "Inception"
        MODEL_VERSION = "V1"


        storage_upload(BUCKET_NAME, MODEL_NAME, MODEL_VERSION, rm=False)

        print("Model has been trained and uploaded to GCP")

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)




if __name__ == "__main__":
    trainer = Trainer()

    #Train model on GCP
    # trainer.train_GCP_data(n=1000, pickle_source=True, make_file=True)


    #Train model locally
    trainer.train_local_data(n=1000, pickle_source=True, make_file=True)
