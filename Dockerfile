
# write some code to build your image
FROM python:3.10.3-buster

COPY api /api
COPY requirements.txt /requirements.txt
COPY DoggyDetector/ DoggyDetector
COPY model.joblib/ model.joblib
# Not sure what the lecture means by loading the trained model

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn app.simple:app --host 0.0.0.0
