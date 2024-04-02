import pandas as pd
import numpy as np
from IPython.display import HTML
import random
import re

# Analyzing the data and building some analysis
# Function to load symptoms, diseases, and adjacency matrix
def load_data():
    df = pd.read_excel('symptoms.xlsx')
    # we replace the whitespaces and then the '_' character by an space
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

    adj_mat = np.zeros((len(symptoms),len(diseases)))
    for i in range(len(df)):
        for j in range(1, 18):  # Assuming 17 symptoms columns max
            disease = df.iloc[i,0]
            symptom = df.iloc[i,j]
            if pd.notnull(symptom):
                symptom = symptom.strip()  # strip leading and trailing whitespace (after and before each symptom names)
                dis_index = diseases.index(disease)
                sym_index = symptoms.index(symptom)
                adj_mat[sym_index, dis_index] += 1

    return symptoms, diseases, adj_mat

# Function to load descriptions and precautions
def load_descriptions():
    df_desc = pd.read_excel('symptoms.xlsx', sheet_name='symptom_Description')
    return df_desc

def load_precautions():
    df_prec = pd.read_excel('symptoms.xlsx', sheet_name='symptom_precaution')
    return df_prec

# Naives Bayes Classifier
non_diagnosis_responses = ['I cannot give you a possible diagnosis','Please, try it again','Please, give me more details','I do not understand what you mean']

# Bayesian classifier
def bayesian_classifier(adj_mat, symptom_list, symptoms, diseases):
    
    # Use re.sub() to remove the special characters from each symptom in the symptom_list
    cleaned_symptom_list = [re.sub(r'[:;¿?¡!-]', '', s).strip().lower() for s in symptom_list]

    # Convert the cleaned symptom list to indices, assuming the symptoms are found in the cleaned list
    sym = [symptoms.index(s) for s in cleaned_symptom_list if s in symptoms]

    p_dis = adj_mat.sum(axis=0) / adj_mat.sum()
    p_sym = adj_mat.sum(axis=1) / adj_mat.sum()
    dist = []

    for i in range(len(diseases)):
        # computing the bayes probability
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
    print('Recommended precautions:')
    for i in range(1, 5):
        print(f"- {precautions[f'Precaution_{i}']}")

# Print description
def print_description(disease, df_desc):

    desc = df_desc['Disease'].str.lower() == disease.lower()
    if desc.any():
        description = df_desc.loc[desc, 'Description'].iloc[0]
        print(f'{description}\n')
    else:
        pass

# print('Please enter your symptoms separated by commas from the list below:')

# # Get user input and process it
# user_input = input('Enter symptoms: ')
# # remove this special characters r'''[.:;¿?¡!'"=+/\[\]{}()`~@$%^&*\d|]'''
# cleaned_input = re.sub(r'''[.:;¿?¡!\<>'"=+/\[\]{}()`~@$%^&*|\d\\]''', '', user_input)
# cleaned_input = cleaned_input.replace('_', ' ').replace('-', ' ')
# user_symptoms = [sym.strip() for sym in cleaned_input.split(',') if sym.strip() in symptoms]

# # Check if the user entered symptoms that are in the list
# if not user_symptoms:
#     non_diagnosis_responses[random.randrange(4)]
# else:
#     # Call the bayesian_classifier function
#     diagnosis = bayesian_classifier(adj_mat, user_symptoms, symptoms, diseases)
#     print(f'\n{user_symptoms}')
#     print(f'\nThe most likely diagnosis is: {diagnosis}\n')
#     print_description(diagnosis, df_desc)
#     print_precautions(diagnosis, df_prec)
        
if __name__ == "__main__":
    symptoms, diseases, adj_mat = load_data()
    df_desc = load_descriptions()
    df_prec = load_precautions()