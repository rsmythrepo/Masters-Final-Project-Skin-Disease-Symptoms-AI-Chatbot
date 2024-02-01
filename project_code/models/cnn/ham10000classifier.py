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
#from tensorflow.keras import datasets, layers, models
#import keras
#from keras.utils import to_categorical # used for converting labels to one-hot-encoding
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
#from sklearn.model_selection import train_test_split
#from scipy import stats
#from sklearn.preprocessing import LabelEncoder

def print_metadata(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Get the absolute path, move to the root directory and add file path (works for everyone)
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = my_path.replace('\project_code\models\cnn', '')
    full_path = os.path.join(path, "data/raw/HAM10000/HAM10000_metadata.csv")
    with open(full_path) as f:
        ham10000_df = pd.read_csv(f)
    print(ham10000_df)
