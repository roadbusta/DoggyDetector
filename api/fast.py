from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import cv2
from google.cloud import storage
import os
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


@app.get("/predict")
def gcp_predict(storage):
    return {"storage location": storage}


@app.get("/retrieve")
def retrieve_image(BUCKET_NAME, BLOB_NAME):

    try:
        storage_client = storage.Client() #Set the storage_client
        FILE_PATH = os.path.join(os.getcwd(), 'test') #Set the end file path
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)
        with open(FILE_PATH, 'wb') as f:
            storage_client.download_blob_to_file(blob,f)

        return True

    except Exception as e:
        print(e)
        return False





def api_predict(pickup_datetime, pickup_longitude, pickup_latitude,
                dropoff_longitude, dropoff_latitude, passenger_count):

    # Put the inputs into a dictionary
    response_dict = {
        "key": f"{pickup_datetime}.0019",  #The hardcoded value
        "pickup_datetime": pickup_datetime,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count
    }

    # Create a dataframe from a dictionary
    df = pd.DataFrame(response_dict, index=[0])

    #Convert the values into the correct data types
    df.pickup_longitude = df.pickup_longitude.astype('float64')
    df.pickup_latitude = df.pickup_latitude.astype('float64')
    df.dropoff_longitude = df.dropoff_longitude.astype('float64')
    df.dropoff_latitude = df.dropoff_latitude.astype('float64')
    df.passenger_count = df.passenger_count.astype('int64')

    # create a datetime object from the user provided datetime
    pickup_datetime = df.pickup_datetime[0]
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # localize the datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    # convert the datetime to the format expected by the pipeline
    formatted_pickup_datetime = utc_pickup_datetime.strftime(
        "%Y-%m-%d %H:%M:%S UTC")

    # put the data back into the dataframe
    df.pickup_datetime = formatted_pickup_datetime

    # Call the predict function
    X = df  #.drop(columns = 'key')
    print("1")
    model = predict.download_model()
    print("2")
    prediction = predict.make_prediction(model, X)

    return {"prediction": prediction[0]}
