import streamlit as st
import numpy as np
from keras.models import load_model
import json
import random
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from symptoms import bayesian_classifier, print_description, print_precautions, load_data, load_descriptions, load_precautions

symptoms, diseases, adj_mat = load_data()
df_desc = load_descriptions()
df_prec = load_precautions()
non_diagnosis_responses = ['I cannot give you a possible diagnosis','Please, try it again','Please, give me more details','I do not understand what you mean']

# Load your trained model, data, and utility functions
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

# Ww convert the words from sentences to roots
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# it converts the information to 1 and 0, the according if they are in the patterns
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    print(bag)
    return np.array(bag)

# Forecasting the category what each sentence belongs
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res ==np.max(res))[0][0]
    category = classes[max_index]
    return category

# Obtain random responses 
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
            result = random.choice(i['responses'])
            break
    return result

# Streamlit app starts here 
def main():
    st.title("Medical Chatbot")

    # Define a function to handle symptom analysis
    def analyze_symptoms(symptoms_input):
        cleaned_symptoms = re.sub(r'''[.:;¿?¡!\<>'"=+/\[\]{}()`~@$%^&*|\d\\]''', '', symptoms_input)
        cleaned_symptoms = cleaned_symptoms.replace('_', ' ').replace('-', ' ')
        user_symptoms = [sym.strip() for sym in cleaned_symptoms.split(',') if sym.strip() in symptoms]
        
        if user_symptoms:
            diagnosis = bayesian_classifier(adj_mat, user_symptoms, symptoms, diseases)
            description = print_description(diagnosis, df_desc)
            precautions = print_precautions(diagnosis, df_prec)
            return f'\nThe most likely diagnosis is: {diagnosis} \n\nDescription: {description}\n\nPrecautions: {precautions}'
        else:
            return random.choice(non_diagnosis_responses)

    # Check if conversation exists in session state
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    def display_conversation():
        for user_msg, bot_reply in st.session_state.conversation:
            st.text_area("You:", value=user_msg, height=45, key=user_msg[:15] + "_user")
            st.text_area("Bot:", value=bot_reply, height=45, key=bot_reply[:15] + "_bot")

    # Conversation input
    user_input = st.text_input("You: ", key="user_input")

    # Columns for layout
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Submit"):
            predicted_class = predict_class(user_input)
            if predicted_class == "medical_consultation":
                st.session_state['show_symptoms_input'] = True
            else:
                st.session_state['show_symptoms_input'] = False
                response = get_response(predicted_class, intents)
                st.session_state.conversation.append((user_input, "Bot: " + response))
            display_conversation()

    if st.session_state.get('show_symptoms_input'):
        with col2:
            symptoms_input = st.text_input("Enter symptoms separated by commas here:", key="symptoms_input")
            if st.button("Analyze Symptoms", key="analyze"):
                bot_reply = analyze_symptoms(symptoms_input)
                st.session_state.conversation.append((user_input, bot_reply))
                # Reset the state to hide the symptoms input
                st.session_state['show_symptoms_input'] = False
                display_conversation()

    # Always display the conversation at the end
    display_conversation()

if __name__ == "__main__":
    main()