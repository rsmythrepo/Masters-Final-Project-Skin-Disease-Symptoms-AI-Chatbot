from warnings import filterwarnings
filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
np.random.seed(42)
import pickle

import cv2
from sklearn import metrics
from keras.utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.metrics import Precision, Recall
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
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

    return skin_df

''' image pre-processing functions'''
def image_resize(skin_df):
    # Use the path to read images
    skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((32, 32))))
    return skin_df

def image_normalzation(skin_df):
    X = np.asarray(skin_df['image'].tolist())
    images = X / 255.
    return images

def clean_images(images):
    # Assuming 'images' is a list of image arrays
    cleaned_images = [img for img in images if img is not None]
    return cleaned_images

'''further pre-processing'''
def convert_to_grayscale(images):
    grayscale_images = []
    for img in images:
        # Check if the image is in 64-bit floating point format
        if img.dtype == 'float64':
            # Convert the image to 8-bit unsigned integer format
            img = cv2.convertScaleAbs(img)
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale_img_np = np.array(grayscale_img)  # Convert to NumPy array
        grayscale_images.append(grayscale_img_np)
    return grayscale_images

def one_hot_encoding(skin_df):
    # label encoding to numeric values from text - one hot encoding
    le = LabelEncoder()
    le.fit(skin_df['dx'])
    LabelEncoder()
    skin_df['label'] = le.transform(skin_df["dx"])

    Y = skin_df['label']
    labels = to_categorical(Y, num_classes=7)  # Convert to categorical as this is a multiclass classification problem
    return labels

def show_images(images):
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        #plt.xlabel(skin_df['dx'][i])
    plt.show()

def train_and_test_split(images, labels):
    # Features & Labels
    X = images
    Y = labels

    # Split to training and testing
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    return x_train, x_test, y_train, y_test

def performance(history):
    # Plot the training and validation accuracy and loss at each epoch
    loss = history.history['acc']
    val_loss = history.history['val_acc']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training Acc')
    plt.plot(epochs, val_loss, 'r', label='Validation Acc')
    plt.title('Training and validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def equalize_image(image):
    # Ensure the image is in BGR format (assuming it's RGB)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to the grayscale image
    equalized_gray = cv2.equalizeHist(gray_image)

    # Convert the equalized grayscale image back to RGB
    equalized_image = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2RGB)

    return equalized_image






if __name__ == '__main__':

    my_path = os.path.abspath(os.path.dirname(__file__))
    path = my_path.replace('\project_code\models\cnn', '')

    #´Get metadata
    metadata_df = ham10000_metadata(path)
    print(metadata_df.head())

    # Get images
    skin_df = image_ingestion(metadata_df)
    print('images')
    print(skin_df.head())

    # Resize images
    skin_df = image_resize(skin_df)

    # Equalized and convert to greyscale
    #skin_df['equalized_image'] = skin_df['image'].apply(equalize_image)

    print('resized')
    print(skin_df.head())

    # Normalization
    images = image_normalzation(skin_df)
    print(skin_df.head())

    # Convert to grayscale
    #images = convert_to_grayscale(images)

    # Show Images
    show_images(images)

    # Clean images
    #images = clean_images(images)

    # One hot encoding of labels
    labels = one_hot_encoding(skin_df)

    # Get the train and test split
    x_train, x_test, y_train, y_test = train_and_test_split(images, labels)

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
    print(model.compile(loss='categorical_crossentropy',
                        optimizer='Adam',
                        metrics=['acc', Precision(), Recall()]))

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

    with open('skin_cancer_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    performance(history)

    # Print the metrics
    print("Training Accuracy:", history.history['acc'])
    print("Training Precision:", history.history['precision'])
    print("Training Recall:", history.history['recall'])
    #print("Training F1 Score:", history.history['f1_score'])

    print("Validation Accuracy:", history.history['val_acc'])
    print("Validation Precision:", history.history['val_precision'])
    print("Validation Recall:", history.history['val_recall'])
    #print("Validation F1 Score:", history.history['val_f1_score'])

    y_train_predict = model.predict(x_train)

    print("The model performance for training set")
    print("RMSE:")
    print(metrics.mean_squared_error(y_train, y_train_predict, squared=False))
    print("MSE:")
    print(metrics.mean_squared_error(y_train, y_train_predict))
    print("MAE:")
    print(metrics.mean_absolute_error(y_train, y_train_predict))
    print("R2:")
    print(metrics.r2_score(y_train, y_train_predict))

    print("\n")

    y_test_predict = model.predict(x_test)
    print("The model performance for testing set")
    print("RMSE:")
    print(metrics.mean_squared_error(y_test, y_test_predict, squared=False))
    print("MSE:")
    print(metrics.mean_squared_error(y_test, y_test_predict))
    print("MAE:")
    print(metrics.mean_absolute_error(y_test, y_test_predict))
    print("R2:")
    print(metrics.r2_score(y_test, y_test_predict))






