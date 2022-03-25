
# write some code to build your image
FROM python:3.10.3-buster
# Trying a different docker image with inbuilt opencv
# FROM jjanzic/docker-python3-opencv

COPY api /api
COPY requirements.txt /requirements.txt
COPY DoggyDetector /DoggyDetector
COPY model.joblib /model.joblib
COPY breed_list.pickle /breed_list.pickle
COPY doggy-detector-2022-c42f18ed1a2f.json /credentials.json



RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

#Trying to set the python environment variable
RUN export PYTHONPATH="$PYTHONPATH:/DoggyDetector"
RUN export GOOGLE_APPLICATION_CREDENTIALS= "/credentials.json"

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
