
# write some code to build your image
FROM python:3.10.3-buster
# Trying a different docker image with inbuilt opencv
# FROM jjanzic/docker-python3-opencv

COPY api /api
COPY requirements.txt /requirements.txt
COPY DoggyDetector /DoggyDetector
COPY model.joblib /model.joblib
COPY breed_list.pickle /breed_list.pickle



RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

#Trying to set the python environment variable
RUN export PYTHONPATH="$PYTHONPATH:/DoggyDetector"

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
