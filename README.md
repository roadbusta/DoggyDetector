# Data analysis
- Document here the project: DoggyDetector
- Description: Takes in a single image and returns the prediction
- Data Source: Stanford Dog Dataset
- Model: Inception V3 (transfer learning)

Please document the project the better you can.

Workflow:
- Clone project
- Train machine learning model 
- Containerise and deploy

# Startup the project

You will need access to Google Cloud Platform, Docker and Heroku to clone and deploy this project.


The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for DoggyDetector in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/DoggyDetector`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "DoggyDetector"
git remote add origin git@github.com:{group}/DoggyDetector.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
DoggyDetector-run
```

# Install

Go to `https://github.com/{group}/DoggyDetector` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/DoggyDetector.git
cd DoggyDetector
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
DoggyDetector-run
```
Create API using uvicorn

Test API

Deploy to docker locally

Deploy to GCP container registry

Test access to GCP

Deploy DoggyDetector Website to heroku

Some troubleshooting and FAQ

