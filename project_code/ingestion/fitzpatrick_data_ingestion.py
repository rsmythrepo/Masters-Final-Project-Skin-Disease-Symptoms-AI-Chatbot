import sys
import boto3
import requests
import pandas as pd
from io import StringIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

def extract_fitzpatrick_data():
    url = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
    response = requests.get(url)
    if response.status_code == 200:
        content = response.content.decode('utf-8')
        csv_data = StringIO(content)
        df = pd.read_csv(csv_data)
    else:
        print('Failed to retrieve information from URL')
        df = pd.DataFrame()  # Return an empty DataFrame if data retrieval fails
    return df

def upload_image_to_s3(image_url, bucket_name, object_name):
    try:
        s3_client = boto3.client('s3')
        headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}

        response = requests.get(image_url, stream=True, headers=headers)
        
        if response.status_code == 200:
            key = directory_name + '/' + object_name + '.jpg'
            s3_client.upload_fileobj(response.raw, bucket_name, key)
            print(f"Image uploaded to S3 bucket: {bucket_name} with object key: {object_name}")
        else:
            print(f"Image couldn't be retrieved from {image_url}")
    except Exception as e:
        print(f"Error uploading image to S3: {e}")

if __name__ == "__main__":
    bucket_name = sys.argv[1]
    directory_name = "fitzpatrick/day=20240221"

    # Extract data from the DataFrame
    df = extract_fitzpatrick_data()

    if not df.empty:
        # Upload images in parallel
        with ThreadPoolExecutor(max_workers=15) as executor:  # Adjust max_workers as needed
            futures = []

            for index, row in df.iterrows():
                object_name = row['md5hash']
                image_url = row['url']
                futures.append(executor.submit(upload_image_to_s3, image_url, bucket_name, object_name))

            # Wait for all uploads to complete
            for future in futures:
                future.result()
    else:
        print("DataFrame is empty or data retrieval failed.")