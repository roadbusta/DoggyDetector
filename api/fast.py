from DoggyDetector.data import file_from_gcp, pickle_from_gcp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import cv2 #commenting this out to troubleshoot docker
from google.cloud import storage
import os
import pickle
import joblib
from DoggyDetector.predictor import Predictor
#from TaxiFareModel import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


##Creating a root endpoint that will welcome the developers using our API
@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/retrieve")
def gcp_predict(storage):
    return {"storage location": storage}


@app.get("/predict")
def predict_breed(BUCKET_NAME, BLOB_NAME):

    try:

        #Store the image as a temprary file. This may have to be changed to
        # something else in the future where the file is saved
        storage_client = storage.Client() #Set the storage_client

        IMAGE_FILE_PATH = os.path.join(os.getcwd(), 'test') #Set the end file path. The image is saved as 'test'

        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)
        with open(IMAGE_FILE_PATH, 'wb') as f:
            storage_client.download_blob_to_file(blob,f)

        #Load the model if from pickle
        # BUCKET_PICKLE_LOCATION = "models/Inception/V1/model.joblib"
        # model_pickle = pickle_from_gcp(BUCKET_NAME= BUCKET_NAME,
        #                                BUCKET_PICKLE_LOCATION= BUCKET_PICKLE_LOCATION)

        # model = pickle.loads(model_pickle)


        #Load the model if from model.joblib

        BUCKET_MODEL_LOCATION = "models/Inception/V1/model.joblib"
        MODEL_FILE_PATH = os.path.join(os.getcwd(), 'model.joblib') #This is where it is saved locally
        file_from_gcp(BUCKET_NAME=BUCKET_NAME,
                      BUCKET_PICKLE_LOCATION=BUCKET_MODEL_LOCATION,
                      DESTINATION_FILE_NAME=MODEL_FILE_PATH)


        model = joblib.load(MODEL_FILE_PATH)

        # #Run predict


        predictor = Predictor()

        prediction = predictor.predict(image_path = IMAGE_FILE_PATH, model = model)

        return prediction




    except Exception as e:
        print(e)
        return False
