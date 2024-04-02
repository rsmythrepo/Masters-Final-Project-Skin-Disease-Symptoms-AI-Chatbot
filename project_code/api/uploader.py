from fastapi import APIRouter, status
from project_code.settings import Settings
from project_code.settings import Credentials

import io
import pandas as pd
import psycopg2
import os
from io import StringIO
import json

import zipfile
import boto3


# [x] Upload the raw data in 2 zip from repo
# [x] Upload from 3bucket the zip an put in folder prepare data "with some space for the code"
# [x] Mix the csv and upload on the database 


uploader_db = APIRouter(
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Something is wrong with the request"},
    },
    prefix="/api/uploader_db",
    tags=["uploader_db"],
)

settings = Settings()
db_credentials = Credentials()

s3 = boto3.client('s3', aws_access_key_id=settings.s3_access_key_id,
                   aws_secret_access_key=settings.s3_secret_access_key,
                     aws_session_token=settings.s3_token)

s3_bucket = settings.s3_bucket

user = db_credentials.username
password = db_credentials.password
host = db_credentials.host
port = db_credentials.port
database = db_credentials.database
    
@uploader_db.post("/raw_datasets")
def upload_zip() -> str:
    # choose the path that you want Raphaelle
    raw_data = 'C:/Users/franc/BTS/final project/repositorysmith/Masters-Final-Project-Skin-Disease-Symptoms-AI-Chatbot/data/raw'

    try:
        files = os.listdir(raw_data)
        for file_name in files:
            if file_name.endswith('.zip'):
                file_path = os.path.join(raw_data, file_name)
                s3_file_key = f'raw_data/classified_images/date=27-03-2024/{file_name}'  
                s3.upload_file(file_path, s3_bucket, s3_file_key)
        
        return "Datasets uploaded successfully"
    
    except Exception as e:
        return f"Error uploading zip files: {str(e)}"
    
@uploader_db.post("/raw_metadata")
def upload_metadata() -> str:
    # choose the path that you want Raphaelle
    raw_data = 'C:/Users/franc/BTS/final project/repositorysmith/Masters-Final-Project-Skin-Disease-Symptoms-AI-Chatbot/data/raw'

    try:
        files = os.listdir(raw_data)
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(raw_data, file_name)
                s3_file_key = f'raw_data/metadata/date=27-03-2024/{file_name}'  
                s3.upload_file(file_path, s3_bucket, s3_file_key)
        
        return "Metadata uploaded successfully"
    
    except Exception as e:
        return f"Error uploading metadata: {str(e)}"
    
@uploader_db.post("/raw_symptoms_sheets")
def upload_excel() -> str:
    try:
        file_path = 'C:/Users/franc/BTS/final project/repositorysmith/Masters-Final-Project-Skin-Disease-Symptoms-AI-Chatbot/project_code/models/symptoms-chatbot/symptoms.xlsx'
        excel_data = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in excel_data.items():
            csv_file_path = os.path.join(os.path.dirname(file_path), f"{sheet_name}.csv")
            df.to_csv(csv_file_path, index=False)
            s3_file_key = f'raw_data/symptoms_files/date=27-03-2024/{sheet_name}.csv'
            s3.upload_file(csv_file_path, s3_bucket, s3_file_key)
        
        return "Excel file uploaded successfully"
    except Exception as e:
        return f"Error uploading Excel file: {str(e)}"

@uploader_db.post("/raw_intents")
def upload_intents() -> str:
    # choose the path that you want Raphaelle
    file_path = 'C:/Users/franc/BTS/final project/repositorysmith/Masters-Final-Project-Skin-Disease-Symptoms-AI-Chatbot/project_code/models/symptoms-chatbot/intents.json' 
    try:
        files = os.listdir(file_path)
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(file_path, file_name)
                s3_file_key = f'raw_data/intents/date=27-03-2024/{file_name}'  
                s3.upload_file(file_path, s3_bucket, s3_file_key)
        
        return "Json Data uploaded successfully"
    
    except Exception as e:
        return f"Error uploading json data: {str(e)}"

@uploader_db.post("/models")
def upload_models():
    #upload the folder cnn, llm-chatbot, naive_bayes
    file_path= 'C:/Users/franc/BTS/final project/repositorysmith/Masters-Final-Project-Skin-Disease-Symptoms-AI-Chatbot/project_code/models'
    try:
        folders = ['cnn', 'llm-chatbot', 'naive_bayes']
        for folder in folders:
            folder_path = os.path.join(file_path, folder)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                s3_file_key = f'models/{folder}/{file_name}'
                s3.upload_file(file_path, s3_bucket, s3_file_key)
    except Exception as e:
        return f"Error uploading models: {str(e)}"

@uploader_db.post("/prepare_images")
def prepare_images() -> str:
    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix='raw_data/classified_images/date=27-03-2024/')
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            
            if key.endswith('.zip'):
                response_zip = s3.get_object(Bucket=s3_bucket, Key=key)
                
                zip_file_bytes = response_zip['Body'].read()
                
                with zipfile.ZipFile(io.BytesIO(zip_file_bytes)) as zip_ref:
                    for extracted_file in zip_ref.namelist():
                        file_extension = os.path.splitext(extracted_file)[1].lower()
                        if file_extension in ('.gif', '.png', '.jpg', '.jpeg'):
                            extracted_file_name = os.path.basename(extracted_file)
                            extracted_file_bytes = zip_ref.read(extracted_file)

                            # Add your function here Raphaelle
                            # body = raphaelle_function(extracted_file_bytes)


                            s3_file_key = f'prepared/classified_images/date=27-03-2024/{extracted_file_name}'
                            s3.put_object(Bucket=s3_bucket, Key=s3_file_key, Body=extracted_file_bytes)
        
        return "Data prepared successfully"
    
    except Exception as e:
        return f"Error preparing data: {str(e)}"

@uploader_db.post("/prepare_metadata")
def prepare_metadata() -> str:
    try:
        ddi_filename = 'ddi_metadata.csv'
        fitz_filename = 'fitzpatrick17k.csv'
        folder_name = 'raw_data/metadata/date=27-03-2024/'

        # Retrieve CSV files from S3
        ddi_csv_object = s3.get_object(Bucket=s3_bucket, Key=folder_name + ddi_filename)
        fitz_csv_object = s3.get_object(Bucket=s3_bucket, Key=folder_name + fitz_filename)

        # Read CSV contents
        ddi_csv_content = ddi_csv_object['Body'].read().decode('utf-8')
        fitz_csv_content = fitz_csv_object['Body'].read().decode('utf-8')

        # Convert CSV contents to DataFrame
        df_ddi = pd.read_csv(StringIO(ddi_csv_content))
        df_fitz = pd.read_csv(StringIO(fitz_csv_content))

        df_merged = pd.DataFrame(index=range(len(df_ddi) + len(df_fitz)), columns=["filename", "skin_tone", "malignant"])
        df_merged['filename'] = df_ddi['DDI_file'].tolist() + df_fitz['md5hash'].tolist()
        df_merged['filename'] = df_merged['filename'].apply(lambda x: x + ".jpg" if not x.endswith(".png") else x)

        df_merged['skin_tone'] = df_ddi['skin_tone'].tolist() + df_fitz['fitzpatrick_scale'].tolist()
        df_merged['skin_tone'] = df_merged['skin_tone'].replace([1, 2], 12).replace([3, 4], 34).replace([5, 6], 56)

        df_merged['malignant'] = df_ddi['malignant'].tolist() + df_fitz['three_partition_label'].tolist()
        df_merged['malignant'] = df_merged['malignant'].replace(['malignant'], True).replace(['non-neoplastic', 'benign'], False)

        df_merged = df_merged[df_merged['skin_tone'] != -1]

        folder_prepared = 'prepared/metadata/date=27-03-2024/'

        s3.upload_fileobj(StringIO(df_merged.to_csv(index=False)), s3_bucket, folder_prepared + 'merged_metadata.csv')

        return "Data merged successfully"
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

@uploader_db.post("/metatada_database")
def metadata_database():
    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix='prepared/metadata/date=27-03-2024/')
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.csv'):         
                response_csv = s3.get_object(Bucket=s3_bucket, Key=key)
        
                csv_file_bytes = response_csv['Body'].read()
                df_merged = pd.read_csv(io.BytesIO(csv_file_bytes))

        conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)
        cursor = conn.cursor()

        create_table_query = '''
            CREATE TABLE IF NOT EXISTS metadata (
                filename VARCHAR(255),
                skin_tone INTEGER,
                malignant BOOLEAN
            )
        '''
        cursor.execute(create_table_query)
        conn.commit()

        for index, row in df_merged.iterrows():
            cursor.execute("INSERT INTO metadata (filename, skin_tone, malignant) VALUES (%s, %s, %s)", (row['filename'], row['skin_tone'], row['malignant']))
            conn.commit()
        cursor.close()
        conn.close()

        return "Data store in the database successfully"

    except Exception as e:
        return f"An error occurred: {str(e)}"
    
@uploader_db.post("/symptoms_database")
def symptoms_database():
    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix='raw_data/symptoms_files/date=27-03-2024/')
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key == 'symptoms.csv':         
                response_csv1 = s3.get_object(Bucket=s3_bucket, Key=key)
                csv_file_bytes1 = response_csv1['Body'].read()
                symptoms_df = pd.read_csv(io.BytesIO(csv_file_bytes1))
            elif key == 'symptom_Description.csv':         
                response_csv2 = s3.get_object(Bucket=s3_bucket, Key=key)
                csv_file_bytes2 = response_csv2['Body'].read()
                description_df = pd.read_csv(io.BytesIO(csv_file_bytes2))
            elif key == 'symptom_precaution.csv':
                response_csv3 = s3.get_object(Bucket=s3_bucket, Key=key)
                csv_file_bytes3 = response_csv3['Body'].read()
                precaution_df = pd.read_csv(io.BytesIO(csv_file_bytes3))

        conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)
        cursor = conn.cursor()

        # First table
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS symptoms (
                Disease VARCHAR(255),
                Symptom_1 VARCHAR(255),
                Symptom_2 VARCHAR(255),
                Symptom_3 VARCHAR(255),
                Symptom_4 VARCHAR(255),
                Symptom_5 VARCHAR(255),
                Symptom_6 VARCHAR(255),
                Symptom_7 VARCHAR(255),
                Symptom_8 VARCHAR(255),
                Symptom_9 VARCHAR(255),
                Symptom_10 VARCHAR(255),
                Symptom_11 VARCHAR(255),
                Symptom_12 VARCHAR(255),
                Symptom_13 VARCHAR(255),
                Symptom_14 VARCHAR(255),
                Symptom_15 VARCHAR(255),
                Symptom_16 VARCHAR(255),
                Symptom_17 VARCHAR(255)
            )
        '''
        cursor.execute(create_table_query)
        conn.commit()

        for index, row in symptoms_df.iterrows():
            cursor.execute("INSERT INTO symptoms VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", tuple(row))
            conn.commit()

        # Second table
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS descriptions (
                Disease VARCHAR(255),
                Description TEXT
            )
        '''

        cursor.execute(create_table_query)
        conn.commit()

        for index, row in description_df.iterrows():
            cursor.execute("INSERT INTO descriptions VALUES (%s, %s)", (row['Disease'], row['Description']))
            conn.commit()

        # Third table
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS precautions (
                Disease VARCHAR(255),
                Precaution_1 VARCHAR(255),
                Precaution_2 VARCHAR(255),
                Precaution_3 VARCHAR(255),
                Precaution_4 VARCHAR(255)
            )
        '''

        cursor.execute(create_table_query)
        conn.commit()

        for index, row in precaution_df.iterrows():
            cursor.execute("INSERT INTO precautions VALUES (%s, %s, %s, %s, %s)", tuple(row))
            conn.commit()
            
        cursor.close()
        conn.close()

        return "Data stored in the database successfully"
    
    except Exception as e:
        return f"Error: {str(e)}"

@uploader_db.post("/intents_database")
def intents_database():
    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix='raw_data/intents/date=27-03-2024/')
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json'):         
                response_json = s3.get_object(Bucket=s3_bucket, Key=key)
                json_file_bytes = response_json['Body'].read()
                intents = json.loads(json_file_bytes)

        conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)
        cursor = conn.cursor()

        # Create table for intents
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS intents (
                tag VARCHAR(255),
                patterns JSONB,
                responses JSONB
            )
        '''
        cursor.execute(create_table_query)
        conn.commit()

        # Insert data into intents table
        for intent in intents['intents']:
            cursor.execute("INSERT INTO intents (tag, patterns, responses) VALUES (%s, %s, %s)", (intent['tag'], json.dumps(intent['patterns']), json.dumps(intent['responses'])))
            conn.commit()
            
        cursor.close()
        conn.close()

        return "Intents data stored in the database successfully"
    
    except Exception as e:
        return f"Error: {str(e)}"