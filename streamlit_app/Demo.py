
import pandas as pd
import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from datetime import datetime
import uuid
from symptoms import bayesian_classifier, print_description, print_precautions, load_data, load_descriptions, load_precautions

import cv2
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms


st.set_page_config(
    page_title="DermaChat",
    layout='wide'
)
st.title("DermaChat")
col1, col2 = st.columns([1,1])

# Col1 Left pane image processing
col1.markdown("### Images")
col1.write('Upload a skin lesion image for classification')

# Side bar functionality
st.sidebar.success("Select a page above.")

# Export Button
st.sidebar.button('Export Report')
#Todo add export report functionalities here

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Adjust the size according to your model's input size
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # Transpose the image to (C, H, W) format
    return img

# Create the Xception model with pre-trained weights using the updated model name
model = timm.create_model('legacy_xception', pretrained=False)

# Modify the classifier for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Load the state dictionary from the saved checkpoint
checkpoint = torch.load('../project_code/models/cnn/Xception_model.pth')
model.load_state_dict(checkpoint)


# Class labels
class_labels = ['True', 'False']  # Update with your class labels

# add a button to upload an image
uploaded_file = col1.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:

    # Convert the file to opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    col1.image(img, caption='Uploaded Image', use_column_width=False)


    # Preprocess the input image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((224, 224)),  # Resize to match the input size of your model
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        image = transform(image)
        return image.unsqueeze(0)  # Add batch dimension


    # Perform inference
    def predict(image):
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            model.eval()
            output = model(image_tensor)
            probabilities = torch.sigmoid(output)[0]  # Probabilities for both classes
            prob_true = probabilities[1].item()  # Probability for the positive class (true)
            prob_false = probabilities[0].item()  # Probability for the negative class (false)
            predicted_class = "DermaChat has detected Melanoma." if prob_true > 0.5 else "DermaChat has detected little to no signs of Melanoma."  # ToDo Adjust the threshold ??
        return predicted_class, prob_true, prob_false


    # Example usage
    predicted_class, prob_true, prob_false = predict(img)

    # Concatenating strings and variables into a single string
    predicted_class_text = str(predicted_class)
    prob_true_text = "The probability of the input skin image being classified as malignant is: " + str(prob_true)
    #prob_false_text = "Probability of False: " + str(prob_false)

    # Writing the concatenated strings to the column
    #col1 = st.sidebar
    col1.write(predicted_class_text)
    col1.write(prob_true_text)



# Col2 - Right pane - Chatbot
# ToDo Chatbot code here - instead of st. use col2. to print on the left pane
col2.markdown("### Chatbot")

# Load data
symptoms, diseases, adj_mat = load_data()
df_desc = load_descriptions()
df_prec = load_precautions()
# Non-diagnosis responses
non_diagnosis_responses = ['I cannot give you a possible diagnosis', 'Please, try it again',
                           'Please, give me more details', 'I do not understand what you mean']

# Initialize lemmatizer, load data, and models
# Load intents.json at the beginning of your script
with open('../data/processed/chatbot/intents.json', 'r') as file:
    intents_json = json.load(file)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('../data/processed/chatbot/intents.json').read())
words = pickle.load(open('../project_code/models/naives-bayes/words.pkl', 'rb'))
classes = pickle.load(open('../project_code/models/naives-bayes/classes.pkl', 'rb'))
model = load_model('../project_code/models/llm-chatbot/chatbot_model.h5')


# Clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# Bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Predict class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category


# Get response
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            break
    return result


# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('../data/processed/chatbot/symptoms.xlsx')
    # Replace whitespaces and '_' with spaces
    for col in df.columns:
        if 'Symptom' in col:
            df[col] = df[col].str.replace(' ', '').str.replace('_', ' ')
    symptom_freqs = df.iloc[:, 1:].stack().value_counts()
    symptom_freqs = pd.DataFrame(symptom_freqs)
    symptom_freqs.index.name = 'Symptom'
    symptom_freqs = symptom_freqs.reset_index()
    symptom_freqs = symptom_freqs.rename(columns={'count': 'frequency'})

    symptoms = list(symptom_freqs['Symptom'].unique())
    diseases = list(df['Disease'].unique())

    adj_mat = np.zeros((len(symptoms), len(diseases)))
    for i in range(len(df)):
        for j in range(1, 18):  # Assuming 17 symptoms columns max
            disease = df.iloc[i, 0]
            symptom = df.iloc[i, j]
            if pd.notnull(symptom):
                symptom = symptom.strip()  # Strip leading and trailing whitespace
                dis_index = diseases.index(disease)
                sym_index = symptoms.index(symptom)
                adj_mat[sym_index, dis_index] += 1

    return symptoms, diseases, adj_mat


# Load descriptions and precautions
@st.cache_data
def load_descriptions():
    df_desc = pd.read_excel('../data/processed/chatbot/symptoms.xlsx', sheet_name='symptom_Description')
    return df_desc


@st.cache_data
def load_precautions():
    df_prec = pd.read_excel('../data/processed/chatbot/symptoms.xlsx', sheet_name='symptom_precaution')
    return df_prec


def print_precautions(disease, df_prec):
    # Normalize the case for comparison
    disease = disease.lower()
    # Filter the dataframe for the matching disease
    matching_precautions = df_prec[df_prec['Disease'].str.lower() == disease]
    # Check if there are any matches
    if not matching_precautions.empty:
        precautions = matching_precautions.iloc[0]
        st.write('Recommended precautions:')
        for i in range(1, 5):
            if pd.notnull(precautions[f'Precaution_{i}']):
                st.write(f"- {precautions[f'Precaution_{i}']}")
    else:
        st.write('No precautions available for this disease.')


def print_description(disease, df_desc):
    desc = df_desc['Disease'].str.lower() == disease.lower()
    if desc.any():
        description = df_desc.loc[desc, 'Description'].iloc[0]
        st.write(f'{description}')
    else:
        st.write('No description available for this disease.')


# Process symptoms
def process_symptoms(selected_symptoms):
    # Generate a string from the list of selected symptoms
    user_symptoms_input = ', '.join(selected_symptoms)

    # Ensure only valid symptoms are processed
    user_symptoms = [sym.strip() for sym in selected_symptoms if sym.strip() in symptoms]

    if not user_symptoms:
        # If no valid symptoms are provided, respond with a prompt for more information
        add_to_conversation("Bot", "Please provide your symptoms for a diagnosis.")
    else:
        # Generate diagnosis from valid symptoms
        if user_symptoms:
            diagnosis = bayesian_classifier(adj_mat, user_symptoms, symptoms, diseases)
            if isinstance(diagnosis, str):
                description = print_description(diagnosis, df_desc)
                precautions = print_precautions(diagnosis, df_prec)
            else:
                add_to_conversation('Bot', "Could not determine the diagnosis.")

        # Add diagnosis and information to the conversation
        response = f"The most likely diagnosis is: {diagnosis}\n\n{description}\n\nRecommended precautions:\n{precautions}"
        add_to_conversation("Bot", response)

    # After processing, clear the selection to prevent re-submission
    st.session_state.selected_symptoms = []


# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []


# Add message to conversation history
def add_to_conversation(speaker, message):
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    st.session_state.conversation.append((speaker, message))


# Display the conversation history
def display_conversation():
    for speaker, message in st.session_state.conversation:
        st.text_area(speaker, value=message, height=70, key=uuid.uuid4(), disabled=True)


# Initialize session state variables
if 'user_message' not in st.session_state:
    st.session_state.user_message = ''
if 'user_symptoms_input' not in st.session_state:
    st.session_state.user_symptoms_input = ''
# if 'medical_consultation' not in st.session_state:
#     st.session_state.medical_consultation = False
if 'clear_symptoms' not in st.session_state:
    st.session_state.clear_symptoms = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''


# Function to handle user messages and bot responses
def handle_message():
    user_message = st.session_state.user_input
    if user_message.strip() != '':
        # Save the user message in the conversation
        add_to_conversation("You", user_message)

        # Predict class
        predicted_class = predict_class(user_message)
        if predicted_class == "medical_consultation":
            st.session_state.medical_consultation = True
        else:
            # Get a response and add it to the conversation
            response = get_response(predicted_class, intents_json)
            add_to_conversation("Bot", response)

        # Clear the input box after the message is handled by resetting the state variable.
        st.session_state.user_input = ''


if 'medical_consultation' not in st.session_state:
    st.session_state.medical_consultation = False


# Callback function to clear the symptoms input
def clear_symptoms_input():
    if 'clear_symptoms' in st.session_state and st.session_state.clear_symptoms:
        st.session_state.user_symptoms_input = ''
        st.session_state.clear_symptoms = False


with col2:
    display_conversation()
    col2.markdown("### Chatbot")
    user_input = st.text_input("You:", key="user_input", on_change=handle_message, value=st.session_state.user_input)

    # Trigger handling the message when the user presses Enter or the 'Send' button
    if st.button('Send', key='send_button'):
        handle_message()

    # Symptom selection and submission
    if st.session_state.get('medical_consultation', False):
        symptoms, _, _ = load_data()
        selected_symptoms = st.multiselect("Select your symptoms:", symptoms, key="selected_symptoms")
        if st.button("Submit Symptoms", key="submit_symptoms"):
            if len(selected_symptoms) > 1:  # Ensure at least two symptoms are selected
                process_symptoms(selected_symptoms)
                st.session_state.medical_consultation = False  # Reset the flag
            else:
                col2.write("Please select at least two symptoms.")







