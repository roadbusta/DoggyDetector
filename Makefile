# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* DoggyDetector/*.py

black:
	@black scripts/* DoggyDetector/*.py

test:
	@coverage run -m pytest tests/test_*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr DoggyDetector-*.dist-info
	@rm -fr DoggyDetector.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ----------------------------------
#      Train Locally
# ----------------------------------

# the name of the package inside of our packaged project containing the code that will handle the data and train the mode
PACKAGE_NAME=DoggyDetector

#the main code file of the package for the training
FILENAME = trainer

train_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}



# ----------------------------------
#      Predict Locally
# ----------------------------------

# the name of the package inside of our packaged project containing the code that will handle the data and train the mode
PACKAGE_NAME=DoggyDetector

#the main code file of the package for the training
FILENAME = predictor

predict_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}


# ----------------------------------
#      Google Cloud Platform
# ----------------------------------
# project id - replace with your GCP project id
PROJECT_ID=doggy-detector-2022

# bucket name - replace with your GCP bucket name
BUCKET_NAME=doggy-detector-2022-bucket-v2

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=australia-southeast1


set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

LOCAL_PATH="/Users/joe/code/roadbusta/DoggyDetector/raw_data/Images"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

# the version of Python to be used for the training
PYTHON_VERSION=3.7

FRAMEWORK=scikit-learn

# the version of the machine learning libraries provided by GCP for the training
RUNTIME_VERSION=1.15

BUCKET_TRAINING_FOLDER = training

JOB_NAME=doggy_detector_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')

upload_data:
	@gsutil cp -r ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}


gcp_submit_training:
	@gcloud ai-platform jobs submit training ${JOB_NAME} \
	--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER}  \
	--package-path ${PACKAGE_NAME} \
	--module-name ${PACKAGE_NAME}.${FILENAME} \
	--python-version=${PYTHON_VERSION} \
	--runtime-version=${RUNTIME_VERSION} \
	--region ${REGION} \
	--stream-logs

run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload
