from DoggyDetector.data import category_list, breed_list, create_training_data, data_from_pickle, data_to_pickle
import os
import glob

# ### category_list tests ###
def test_len_category_list():
    #Check that the list is non-zero length
    assert len(category_list(DATADIR="raw_data/Images")) != 0

def test_data_type_category_list():
    #Check that the contains strings
    categories = category_list(DATADIR="raw_data/Images")
    assert type(categories[0]) == str


# ### breed_list tests ###
def test_len_breed_list():
    #Check that the list is non-zero length
    assert len(breed_list(DATADIR="raw_data/Images")) != 0

def test_data_type_breed_list():
    #Check that the contains strings
    breed = breed_list(DATADIR="raw_data/Images")
    assert type(breed[0]) == str

### Training data tests ###

#Note that these tests take some time to validate,
# consider commenting out this code in future builds

def test_len_RGB_create_training_data():

    #Check that X contains at contains image information
    CATEGORIES = category_list(DATADIR="raw_data/Images")
    (X, y) = create_training_data(CATEGORIES, DATADIR="raw_data/Images")
    assert len(X[0][0]) > 0

def test_len_img_create_training_data():
    #Check that X contains at contains image information
    CATEGORIES = category_list(DATADIR="raw_data/Images")
    (X, y) = create_training_data(CATEGORIES, DATADIR="raw_data/Images")
    assert len(X[1][0]) > 0

### Testing data to pickle ###

def test_data_to_pickle():
    pickle_path = "DoggyDetector/data/Pickle Files/Test/"

    # Remove existing pickle files
    files = os.listdir(pickle_path)
    for item in files:
        if item.endswith(".pickle"):
            os.remove(os.path.join(pickle_path, item))

    # Create pickle files
    X = "test x"
    y = "test y"
    data_to_pickle(X, y, pickle_path)

    # Check that there are existing files in the folder
    pickle_files = pickle_path + "*.pickle"
    assert len(glob.glob(pickle_files)) != 0

### Testing data to pickle ###
def test_data_from_pickle():
    pickle_path = "DoggyDetector/data/Pickle Files/Test/"
    X, y = data_from_pickle(pickle_path= pickle_path)

    assert len(X) and len(y) != 0





### Testing model to pickle ###

### Testing pickle to model ###
