from fastapi import APIRouter, status
from project_code.settings import Settings
import os
import requests
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

s3_client = boto3.client('s3')
s3_bucket = 'your_s3_bucket_name'
google_drive_folder_link = 'https://drive.google.com/drive/u/0/folders/18YXxqAvR8wrqcWBxjoZehlJSLGm2wg1W'

def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download file from Google Drive: {response.status_code}")


#fizpatrick dataset has jpg and a csv
#ddi dataset has png and a csv


@uploader_db.post("/take_datasets/{google_drive_folder_link}")
def upload_to_s3(google_drive_folder_link):
    # Extract file IDs from Google Drive folder link
    folder_id = google_drive_folder_link.split('/')[-1]
    url = f"https://www.googleapis.com/drive/v3/files/{folder_id}/children"
    headers = {'Authorization': 'Bearer YOUR_ACCESS_TOKEN'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        files = response.json().get('files', [])
        for file_info in files:
            file_id = file_info['id']
            file_name = file_info['name']
            destination = f"/tmp/{file_name}"
            download_file_from_google_drive(file_id, destination)
            
            # Upload downloaded file to S3 bucket
            try:
                s3_client.upload_file(destination, s3_bucket, file_name)
            except FileNotFoundError:
                return {"message": f"File not found: {destination}"}
            except NoCredentialsError:
                return {"message": "Credentials not available."}

        return {"message": "Datasets uploaded to S3 successfully"}
    else:
        return {"message": f"Failed to retrieve files from Google Drive folder: {response.status_code}"}

# Example usage:

google_drive_folder_link = 'https://drive.google.com/drive/u/0/folders/18YXxqAvR8wrqcWBxjoZehlJSLGm2wg1W'

result = upload_to_s3(google_drive_folder_link, s3_client, s3_bucket)
print(result)

