import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize lemmatizer, load data, and models
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

# Predict class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res ==np.max(res))[0][0]
    category = classes[max_index]
    return category

# Get response
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
            result = random.choice(i['responses'])
            break
    return result

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('symptoms.xlsx')
    # Replace whitespaces and '_' with spaces
    for col in df.columns:
        if 'Symptom' in col:
            df[col] = df[col].str.replace(' ', '').str.replace('_', ' ')
    symptom_freqs = df.iloc[:,1:].stack().value_counts()
    symptom_freqs = pd.DataFrame(symptom_freqs) 
    symptom_freqs.index.name = 'Symptom'
    symptom_freqs = symptom_freqs.reset_index() 
    symptom_freqs = symptom_freqs.rename(columns={'count':'frequency'})

    symptoms = list(symptom_freqs['Symptom'].unique())
    diseases = list(df['Disease'].unique())

    adj_mat = np.zeros((len(symptoms), len(diseases)))
    for i in range(len(df)):
        for j in range(1, 18):  # Assuming 17 symptoms columns max
            disease = df.iloc[i,0]
            symptom = df.iloc[i,j]
            if pd.notnull(symptom):
                symptom = symptom.strip()  # Strip leading and trailing whitespace
                dis_index = diseases.index(disease)
                sym_index = symptoms.index(symptom)
                adj_mat[sym_index, dis_index] += 1

    return symptoms, diseases, adj_mat

# Load descriptions and precautions
@st.cache_data
def load_descriptions():
    df_desc = pd.read_excel('symptoms.xlsx', sheet_name='symptom_Description')
    return df_desc

@st.cache_data
def load_precautions():
    df_prec = pd.read_excel('symptoms.xlsx', sheet_name='symptom_precaution')
    return df_prec

# Non-diagnosis responses
non_diagnosis_responses = ['I cannot give you a possible diagnosis', 'Please, try it again', 'Please, give me more details', 'I do not understand what you mean']

# Bayesian classifier
def bayesian_classifier(adj_mat, symptom_list, symptoms, diseases):
    # Remove special characters from symptoms
    cleaned_symptom_list = [re.sub(r'[:;¿?¡!-]', '', s).strip().lower() for s in symptom_list]

    # Convert cleaned symptoms to indices
    sym = [symptoms.index(s) for s in cleaned_symptom_list if s in symptoms]

    p_dis = adj_mat.sum(axis=0) / adj_mat.sum()
    p_sym = adj_mat.sum(axis=1) / adj_mat.sum()
    dist = []

    for i in range(len(diseases)):
        # Compute the Bayes probability
        prob = np.prod((adj_mat[:,i] / adj_mat[:,i].sum())[sym]) * p_dis[i] / np.prod(p_sym[sym])
        dist.append(prob)
    
    if sum(dist) == 0:
        return non_diagnosis_responses[random.randrange(4)]
    else:
        idx = dist.index(max(dist))
        return diseases[idx]

# Print precautions
def print_precautions(diseases, df_prec):
    precautions = df_prec[df_prec['Disease'].str.lower() == diseases.lower()].iloc[0]
    st.write('Recommended precautions:')
    for i in range(1, 5):
        st.write(f"- {precautions[f'Precaution_{i}']}")

# Print description
def print_description(disease, df_desc):
    desc = df_desc['Disease'].str.lower() == disease.lower()
    if desc.any():
        description = df_desc.loc[desc, 'Description'].iloc[0]
        st.write(f'{description}\n')
    else:
        pass

# Process symptoms
def process_symptoms(user_symptoms_input):
    cleaned_symptoms = re.sub(r'''[.:;¿?¡!\'"<>=+/\[\]{}()`~@$%^&*|\d\\]''', '', user_symptoms_input)
    cleaned_symptoms = cleaned_symptoms.replace('_', ' ').replace('-', ' ')
    user_symptoms = [sym.strip() for sym in cleaned_symptoms.split(',') if sym.strip() in symptoms]

    if user_symptoms:
        confirmation_message = f"OK, your symptoms are: {', '.join(user_symptoms)}. Right?"
        add_to_conversation("Bot", confirmation_message)

        if st.button("Yes"):
            diagnosis = bayesian_classifier(adj_mat, user_symptoms, symptoms, diseases)
            description = print_description(diagnosis, df_desc)
            precautions = print_precautions(diagnosis, df_prec)
            add_to_conversation("Bot", f"The most likely diagnosis is: {diagnosis}\n\n{description}\n\nRecommended precautions:\n{precautions}")
        elif st.button("No"):
            add_to_conversation("Bot", "Let's try again. Please enter your symptoms.")
    else:
        add_to_conversation("Bot", random.choice(non_diagnosis_responses))

# Load data
symptoms, diseases, adj_mat = load_data()
df_desc = load_descriptions()
df_prec = load_precautions()

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Add message to conversation history
def add_to_conversation(speaker, message):
    st.session_state.conversation.append((speaker, message))

# Display conversation history
def display_conversation():
    for index, (speaker, message) in enumerate(st.session_state.conversation):
        key = f"{speaker}_{index}"  # Unique key for each message
        if speaker == "You":
            st.text_area("You:", value=message, height=25, disabled=True, key=key)
        else:
            st.text_area("Bot:", value=message, height=25, disabled=True, key=key)


## Streamlit app starts here
st.title('Medical Chatbot')

# Input from the user
user_message = st.text_input("You:", key="user_input")

if user_message:
    # Save the user message in the conversation
    add_to_conversation("You", user_message)

    # Predict class
    predicted_class = predict_class(user_message)
    
    if predicted_class == "medical_consultation":
        # Ask for symptoms
        st.write("Please enter your symptoms separated by commas:")
        user_symptoms_input = st.text_input("Symptoms:", key="symptoms_input")
        
        if user_symptoms_input:
            process_symptoms(user_symptoms_input)
    else:
        response = get_response(predicted_class, intents)
        add_to_conversation("Bot", response)

# Display the conversation history
display_conversation()