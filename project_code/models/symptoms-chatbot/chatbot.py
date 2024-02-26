import random
import json
import pickle
import pandas as pd
import numpy as np
import re

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model
from symptoms import bayesian_classifier, print_description, print_precautions, load_data, load_descriptions, load_precautions

symptoms, diseases, adj_mat = load_data()
df_desc = load_descriptions()
df_prec = load_precautions()

lemmatizer = WordNetLemmatizer()

#Importamos los archivos generados en el código anterior
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

#Pasamos las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    print(bag)
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res ==np.max(res))[0][0]
    category = classes[max_index]
    return category

# Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
            result = random.choice(i['responses'])
            break
    return result

non_diagnosis_responses = ['I cannot give you a possible diagnosis','Please, try it again','Please, give me more details','I do not understand what you mean']

while True:
    message = input("You: ")
    if message.lower() == "quit":
        break

    predicted_class = predict_class(message)
    if predicted_class == "medical_consultation":
        while True:
            print("Please enter your symptoms separated by commas:")
            user_symptoms_input = input()
            # Process and analyze symptoms here...
            cleaned_symptoms = re.sub(r'''[.:;¿?¡!\<>'"=+/\[\]{}()`~@$%^&*|\d\\]''', '', user_symptoms_input)
            cleaned_symptoms = cleaned_symptoms.replace('_', ' ').replace('-', ' ')
            user_symptoms = [sym.strip() for sym in cleaned_symptoms.split(',') if sym.strip() in symptoms]

            # Ask for user validation
            if user_symptoms:
                print(f"OK, your symptoms are: {', '.join(user_symptoms)}. Right? In case it is correct, please write Yes. In case I am missing something say no")
                validation = input()
                if validation.lower() == "yes":
                    break  # Exit the loop if symptoms are validated
                else:
                    print("Let's try again. Please enter your symptoms.")

            else:
                print(random.choice(non_diagnosis_responses))

        diagnosis = bayesian_classifier(adj_mat, user_symptoms, symptoms, diseases)
        print(f'\nThe most likely diagnosis is: {diagnosis}\n')
        print_description(diagnosis, df_desc)
        print_precautions(diagnosis, df_prec)
    else:
        response = get_response(predicted_class, intents)
        print("Bot:", response)
