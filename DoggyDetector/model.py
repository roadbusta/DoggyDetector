from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense



def init_model(num_classes):
    """
    Creates the final layers of the model
    """

    model = Sequential()
    #model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy', metrics=['accuracy'])

    return model
