import random
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2

import nltk
from nltk.stem import WordNetLemmatizer # To convert words to their root form

'''For creating the neural network'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

'Set the random seed for reproducibility'
SEED = 5
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('augmented_intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?','!','¡','¿','.',',',':',';','/','|','-','+','@','%']

'''Classify patterns and categories'''
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

'''It passes the information to ones or zeros, the according with the words in each category to make the train'''
training = []
test_x = []
test_y = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
    test_x.append(bag)
    test_y.append(output_row)

# Shuffle the training data
# random.shuffle(training)
# training = np.array(training, dtype=object) 
# print(training) 

'''Split the data into training and testing sets'''
train_x, test_x, train_y, test_y = train_test_split(
    np.array([i[0] for i in training]), 
    np.array([i[1] for i in training]), 
    test_size=0.2,  # 80% training, 20% testing
    random_state=5
)

'''Further split the training data into training and validation sets'''
train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, 
    train_y, 
    test_size=0.25,  # 60% training, 20% validation
    random_state=5
)

'''Creating the neural network '''
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu', kernel_regularizer=l2(0.001)))       # neurons
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))                                       # layers
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))                                        # layers
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))                                        # layers
model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))                                        # layers
# model.add(Dropout(0.2))
model.add(Dense(len(train_y[0]), activation='softmax'))

'''Creating the optimizer and compile it'''
optimizer = Adam(learning_rate=0.0003, weight_decay = 0.001)
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

'''Training the model and save it'''
train_process = model.fit(
    train_x,
    train_y,
    epochs=1000,
    batch_size=64,
    verbose=1,
    validation_data=(valid_x, valid_y), 
    callbacks=[early_stopping],
    shuffle=False
)
model.save("chatbot_seq_bow.h5", train_process)

'''testing the model
Also, it is essential to evaluate the model's performance on a separate test set to get a better understanding of its generalization capabilities
'''
''' Train set'''
train_loss, train_accuracy = model.evaluate(train_x, train_y)

print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)

''' Test set'''
test_loss, test_accuracy = model.evaluate(test_x, test_y)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

'''Predict the labels for the test set'''
y_pred = model.predict(test_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_y, axis=1)

report = classification_report(y_true, y_pred_classes, target_names=classes)
print(report)
