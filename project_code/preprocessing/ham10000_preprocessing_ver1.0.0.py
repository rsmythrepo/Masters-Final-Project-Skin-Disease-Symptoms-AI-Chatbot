import boto3
import zipfile
import io
from bs4 import BeautifulSoup

""" Web scrapping zip files from their original site """
# To Do
# Web Scraping:
url = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Finding the link to the zip file
zip_file_url = None
for link in soup.find_all('a'):
    href = link.get('href')
    if href.endswith('.zip'):
        zip_file_url = href
        break

if zip_file_url:
    # Downloading the Zip File
    zip_file_response = requests.get(zip_file_url)
    zip_file_data = zip_file_response.content

""" Stream data from main website to S3 """
# To Do
s3 = boto3.client('s3')

def upload_files_from_zip(zip_file_path, bucket_name):
    with open(zip_file_path, 'rb') as f:
        with zipfile.ZipFile(f, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                with zip_ref.open(file_info.filename) as file_in_zip:
                    s3.put_object(Bucket=bucket_name, Key=file_info.filename, Body=file_in_zip)

#To Do

""" Verify that all images have metadata """
# To Do

""" """