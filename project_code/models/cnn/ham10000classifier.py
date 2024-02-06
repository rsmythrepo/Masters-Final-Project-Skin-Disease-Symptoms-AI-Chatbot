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

# import cv2 as cv
#import keras
# from tensorflow.keras import datasets, layers, models
#from keras.utils import to_categorical # used for converting labels to one-hot-encoding
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
#from sklearn.model_selection import train_test_split
#from scipy import stats
#from sklearn.preprocessing import LabelEncoder

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



if __name__ == '__main__':

    my_path = os.path.abspath(os.path.dirname(__file__))
    path = my_path.replace('\project_code\models\cnn', '')

    #´Get metadata
    metadata_df = ham10000_metadata(path)
    print(metadata_df.head())

    # Get images
    skin_df = image_ingestion(metadata_df)
    print(skin_df.head())

    # TODO add the numerical labels per classifier




