from fastapi import APIRouter, status
from project_code.settings import Settings
from project_code.settings import Credentials

import io
import pandas as pd
import psycopg2
from psycopg2 import sql
import numpy as np
import base64
from PIL import Image
import os

import zipfile
import boto3
from botocore.exceptions import NoCredentialsError

# [] Mix the csv and upload on the database 
# [] Upload the raw data in 2 zip from repo
# [] Upload from 3bucket the zip an put in folder prepare data "with some space for the code"



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
            if file_name.endswith('.zip'):
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

                            # Add your function here
                            # body = raphaelle_function(extracted_file_bytes)


                            s3_file_key = f'prepared_data/{extracted_file_name}'
                            s3.put_object(Bucket=s3_bucket, Key=s3_file_key, Body=extracted_file_bytes)
        
        return "Data prepared successfully"
    
    except Exception as e:
        return f"Error preparing data: {str(e)}"


@uploader_db.post("/merge_metadata")
def merge_metadata() -> str:
    pass