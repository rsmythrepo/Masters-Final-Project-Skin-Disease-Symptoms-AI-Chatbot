from fastapi import APIRouter, status
from project_code.settings import Settings
from project_code.settings import Credentials

import io
import pandas as pd
import psycopg2
import os
from io import StringIO

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
    
@uploader_db.post("/upload_raw_data")
def upload_data() -> str:
    # choose the path that you want Raphaelle
    raw_data = 'C:/Users/franc/BTS/final project/repositorysmith/Masters-Final-Project-Skin-Disease-Symptoms-AI-Chatbot/data/raw'

    try:
        files = os.listdir(raw_data)
        for file_name in files:
            file_path = os.path.join(raw_data, file_name)
            s3_file_key = f'raw_data/{file_name}'  
            s3.upload_file(file_path, s3_bucket, s3_file_key)
        
        return "Data uploaded successfully"
    
    except Exception as e:
        return f"Error uploading data: {str(e)}"

@uploader_db.post("/prepare_data")
def prepare_data() -> str:
    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix='raw_data/')
        
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


                            s3_file_key = f'prepared_data/{extracted_file_name}'
                            s3.put_object(Bucket=s3_bucket, Key=s3_file_key, Body=extracted_file_bytes)
        
        return "Data prepared successfully"
    
    except Exception as e:
        return f"Error preparing data: {str(e)}"


    
@uploader_db.post("/merge_and_upload_db")
def create_the_database():
    try:
        ddi_filename = 'ddi_metadata.csv'
        fitz_filename = 'fitzpatrick17k.csv'
        folder_name = 'raw_data/'

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

        return "Data merged and uploaded to database successfully"

    except Exception as e:
        return f"An error occurred: {str(e)}"