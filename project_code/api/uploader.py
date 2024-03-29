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

import zipfile
import boto3
from botocore.exceptions import NoCredentialsError



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

def png_to_numpy_array(file_path):
    with Image.open(file_path) as img:
        img_array = np.array(img)
    return img_array

@uploader_db.post("/add_table_DDI")
def prepare_data_DDI() -> str:
    try:
        # Step 1: Download and unzip the file from S3
        response = s3.get_object(Bucket=s3_bucket, Key='skin_disease_databases/DDI-20240329T152040Z-001.zip')
        with zipfile.ZipFile(io.BytesIO(response['Body'].read())) as zip_ref:
            zip_ref.extractall('/tmp')  # Extract to temporary directory
            print("Extracted files:")
            print(zip_ref.namelist())

            # Step 2: Transform images to numpy array
            image_data = {}
            for file_name in zip_ref.namelist():
                if file_name.endswith('.png'):
                    # Remove 'DDI/' from the filename
                    file_name_without_ddi = file_name.split('/')[-1]
                    # Convert image to numpy array and add to dictionary with file name as key
                    numpy_array = png_to_numpy_array('/tmp/' + file_name)  # Provide full path to the image
                    image_data[file_name_without_ddi] = numpy_array
                    print(file_name_without_ddi)

            # Step 3: Read CSV file and create DataFrame
            csv_file = [file_name for file_name in zip_ref.namelist() if file_name.endswith('.csv')][0]
            csv_data = pd.read_csv('/tmp/' + csv_file)  # Provide full path to the CSV file
            #print the head of the csv
            print("Head of the csv file:")
            print(csv_data.head())

            # Step 4: Insert image arrays into the DataFrame
            image_arrays = []
            for file_name in csv_data['DDI_file']:
                if file_name in image_data:
                    image_arrays.append(image_data[file_name])
                else:
                    print(f"Image array not found for file: {file_name}")
                    image_arrays.append(None)  # Insert None if image array not found

            csv_data['image_array'] = image_arrays

            # Step 5: Insert data into the database
            try:
                # Establish connection to PostgreSQL database
                conn = psycopg2.connect(
                    dbname=database,
                    user=user,
                    password=password,
                    host=host,
                    port=port
                )

                # Create a cursor object using the connection
                cursor = conn.cursor()

                # Create a table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS DDI_table (
                        DDI_file VARCHAR(255),
                        skin_tone INTEGER,
                        malignant BOOLEAN,
                        disease VARCHAR(255),
                        image_array BYTEA
                    )
                """)

                # Insert data into the table
                for index, row in csv_data.iterrows():
                    # Get image file path
                    image_file = row['DDI_file']
                    # Get image array
                    image_array = row['image_array']
                    if image_array is not None:
                        # Encode image array as bytes
                        image_bytes = base64.b64encode(image_array).decode('utf-8')
                        cursor.execute(
                            sql.SQL("INSERT INTO DDI_table (DDI_file, skin_tone, malignant, disease, image_array) VALUES (%s, %s, %s, %s, %s)"),
                            (image_file, row['skin_tone'], row['malignant'], row['disease'], image_bytes)
                        )
                    else:
                        print(f"Image array not found for file: {image_file}")

                # Commit the transaction
                conn.commit()

                # Close the cursor and connection
                cursor.close()
                conn.close()

                print("Data inserted successfully")
                return "Data preparation for DDI successful"

            except psycopg2.Error as e:
                print(f"PostgreSQL error occurred: {e}")
                return "Error inserting data into the database"

    except NoCredentialsError:
        return "AWS credentials not found"

    except Exception as e:
        return f"An error occurred: {str(e)}"


