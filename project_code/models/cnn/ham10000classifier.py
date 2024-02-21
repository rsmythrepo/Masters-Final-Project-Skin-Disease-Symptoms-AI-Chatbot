import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
from sklearn.metrics import confusion_matrix

#import cv2 as cv
#import keras
# from tensorflow.keras import datasets, layers, models
from keras.utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
#from scipy import stats
from sklearn.preprocessing import LabelEncoder

def ham10000_metadata(path):

    full_path = os.path.join(path, "data/raw/HAM10000/HAM10000_metadata.csv")
    with open(full_path) as f:
        ham10000_df = pd.read_csv(f)

    return ham10000_df

def image_ingestion(skin_df):
    # Path to data
    my_path = os.path.abspath(os.path.dirname(__file__))
    real_path = my_path.replace('\project_code\models\cnn', '')

    # Ensures the right image is read for the right ID
    image_path = {os.path.splitext(os.path.basename(x))[0]: x
                  for x in glob(os.path.join(real_path, 'data/raw/HAM10000/', '*', '*.jpg'))}

    # Define the path and add as a new column
    skin_df['path'] = skin_df['image_id'].map(image_path.get)

    # Use the path to read images
    skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((32, 32))))

    return skin_df

def show_images(skin_df):
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(skin_df['image'][i], cmap=plt.cm.binary)
        plt.xlabel(skin_df['dx'][i])
    plt.show()

def train_and_test_split(df):
    # Convert the dataframe column of images into numpy array

    X = np.asarray(df['image'].tolist())
    X = X / 255.

    # Assign label values to Y
    Y = df['label']
    Y_cat = to_categorical(Y, num_classes=7)  # Convert to categorical as this is a multiclass classification problem

    # Split to training and testing
    x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    my_path = os.path.abspath(os.path.dirname(__file__))
    path = my_path.replace('\project_code\models\cnn', '')

    #´Get metadata
    metadata_df = ham10000_metadata(path)
    print(metadata_df.head())

    # Get images
    skin_df = image_ingestion(metadata_df)
    print(skin_df.head())

    # Take a first look at the images
    #show_images(skin_df)

    # label encoding to numeric values from text - one hot encoding
    le = LabelEncoder()
    le.fit(skin_df['dx'])
    LabelEncoder()

    print('class names:')
    print(list(le.classes_))

    skin_df['label'] = le.transform(skin_df["dx"])
    print(skin_df)

    # Get the train and test split
    x_train, x_test, y_train, y_test = train_and_test_split(skin_df)

    # Define the model.
    num_classes = 7

    # defining the neural netwrok
    model = Sequential()

    # define the input layer, convolutional layer
    model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(32, 32, 3)))

    # next layer, maxpooling layer, simplifying the result
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # another convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu'))

    # next layer, maxpooling layer, simplifying the result
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # another convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # next layer, maxpooling layer, simplifying the result
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # flattening layer, making it 1 dimensional
    model.add(Flatten())

    # scales all the result so they add up to 1 so that we can get -
    # a distance of probability for the indivisual classifications
    model.add(Dense(32))
    model.add(Dense(7, activation='softmax'))
    model.summary()

    # Complile the model
    print(model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc']))

    # Train and fit the model
    batch_size = 12
    epochs = 50

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=2)

    score = model.evaluate(x_test, y_test)
    print('Test accuracy:', score[1])






