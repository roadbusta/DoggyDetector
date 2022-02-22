from sqlite3 import DatabaseError
from DoggyDetector.utils import category_list, create_training_data



if __name__ == "__main__":
    #Generate images of [IMG_SIZE] pixels
    CATEGORIES = category_list()

    X, y = create_training_data( CATEGORIES)

    print(len(X))
