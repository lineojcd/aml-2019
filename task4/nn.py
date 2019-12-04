from main import read_test_data, read_data, StandardScaler
import sklearn
import numpy
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Conv2D, MaxPooling2D, Flatten
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, to_categorical
from time import gmtime, strftime, localtime, time


def lstm():
    model = Sequential()
    return model


def cnn_paper():
    # implemented the model here
    # https://iopscience.iop.org/article/10.1088/1741-2552/aadc1f/pdf
    # create model
    model = Sequential()
    # feature extraction
    model.add(Reshape((8, 900, 1), input_shape=(7200, 1)))

    model.add(Conv2D(3, (1, 10), padding='same', activation='relu'))

    model.add(Conv2D(3, (3, 1), activation='relu'))
    model.add(MaxPooling2D((2, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(5, (1, 5), padding='same', activation='relu'))

    model.add(Conv2D(5, (3, 1), activation='relu'))
    model.add(MaxPooling2D((1, 5), strides=(1, 3), padding='same'))

    model.add(Conv2D(7, (1, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D((1, 6), strides=(1, 4)))

    model.add(Conv2D(10, (1, 37)))

    model.add(Conv2D(3, (1, 1)))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def cnn():

    model = Sequential()
    # feature extraction
    # model.add(Reshape((1536, 1), input_shape=(1536,)))
    # model.add(Reshape((3, 512, 1), input_shape=(1536, 1)))
    # model.add(Reshape((3, 512, 1), input_shape=(1536,)))

    model.add(Conv2D(3, (1, 10), padding='same', data_format='channels_first',
                     activation='relu', input_shape=(3, 512, 1)))

    model.add(Conv2D(3, (3, 1), activation='relu'))
    model.add(MaxPooling2D((2, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(5, (1, 4), activation='relu', padding='same'))

    model.add(Conv2D(5, (1, 1), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(1, 4), padding='same'))

    model.add(Conv2D(7, (1, 4), padding='same', activation='relu'))
    model.add(MaxPooling2D((1, 6), strides=(1, 4)))

    model.add(Conv2D(10, (1, 15)))

    model.add(Conv2D(3, (1, 1)))

    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = cnn()
    X, y = read_data()
    X = StandardScaler().fit_transform(X)
    y = to_categorical(y)
    X = X.reshape((-1, 3, 512, 1))
    model.fit(X, y, verbose=10)
