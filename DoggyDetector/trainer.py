# Import relevant libraries
from tensorflow.keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from keras.applications import inception_v3
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import matplotlib.pyplot as plt
import math
import pickle

class Trainer():
    def train():
        # Transform X and y as required
        X_in = np.array(X_small)
        num_classes = len(set(y_small))
        # Convert y into categorical data
        y_in = to_categorical(y_small, num_classes)

        # Create the train, test and validation sets
        first_split = int(len(X_small) /6.)
        second_split = first_split + int(len(X_small) * 0.2)
        X_test, X_val, X_train = X_in[:first_split], X_in[first_split:second_split], X_in[second_split:]
        y_test, y_val, y_train = y_in[:first_split], y_in[first_split:second_split], y_in[second_split:]

        # Create a function that converts arrays into tensors
        def array_to_tensor(input_array):
            """
            A function to create arrays into tensors for Keras
            """
            list_of_tensors = [np.expand_dims(image, axis=0) for image in tqdm(input_array)]
            return np.vstack(list_of_tensors)

        # pre-process the data for Keras - Converts to (224, 224) and converts into a numpy array using PIL.
        train_tensors = array_to_tensor(X_train).astype('float32')/255
        val_tensors = array_to_tensor(X_val).astype('float32')/255
        test_tensors = array_to_tensor(X_test).astype('float32')/255


        # Create predict files (?) Note: I'm not exactly sure what this step does exactly
        input_size=224 #Note input size
        num_classes=120
        S=1
        train_len=len(X_train)

        inception_bottleneck=inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        train_i_bf = inception_bottleneck.predict(train_tensors, batch_size=32, verbose=1)
        val_i_bf = inception_bottleneck.predict(val_tensors, batch_size=32, verbose=1)
        test_i_bf = inception_bottleneck.predict(test_tensors, batch_size=32, verbose=1)
        # print('InceptionV3 train bottleneck features shape: {} size: {:,}'.format(train_i_bf.shape, train_i_bf.size))
        # print('InceptionV3 valid bottleneck features shape: {} size: {:,}'.format(val_i_bf.shape, val_i_bf.size))
        # print('InceptionV3 test bottleneck features shape: {} size: {:,}'.format(test_i_bf.shape, test_i_bf.size))

        # Save files (?) Not sure if this is always required
        np.save('bottleneck_features_train_inception.npy', train_i_bf)
        np.save('bottleneck_features_val_inception.npy', val_i_bf)
        np.save('bottleneck_features_test_inception.npy', test_i_bf)

        # load the bottleneck features saved earlier
        train_data = np.load('bottleneck_features_train_inception.npy')
        val_data = np.load('bottleneck_features_val_inception.npy')
        test_data = np.load('bottleneck_features_test_inception.npy')
        batch_size=32
        epochs=50

        model = Sequential()
        #model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_data, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(val_data, y_val))

        #Save the weights
        model.save_weights('inception_model_2.h5')

        #Save the model as a pickle file
        pickle.dump(model, open('model.pkl', 'wb'))

        (eval_loss, eval_accuracy) = model.evaluate(
            val_data, y_val, batch_size=batch_size, verbose=1)

        print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
        print("[INFO] Loss: {}".format(eval_loss))






if __name__ == "__main__":
