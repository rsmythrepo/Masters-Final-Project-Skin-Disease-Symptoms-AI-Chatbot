import os

from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, status

from project_code.settings import Settings

settings = Settings()
BASE_URL = "https://samples.adsbexchange.com/readsb-hist/2023/11/01/"

s1 = APIRouter(
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Something is wrong with the request"},
    },
    prefix="/api/s1",
    tags=["s1"],
)


# example endpoint
@s1.post("/images/download")
def download_data() -> str:
    """Downloads the **first 1000** files AS IS inside the folder data/20231101

    data: https://samples.adsbexchange.com/readsb-hist/2023/11/01/
    documentation: https://www.adsbexchange.com/version-2-api-wip/
        See "Trace File Fields" section

    Think about the way you organize the information inside the folder
    and the level of preprocessing you might need.

    To manipulate the data use any library you feel comfortable with.
    Just make sure to configure it in the `pyproject.toml` file
    so it can be installed using `poetry update`.
    """

    # Using python requests library for downloads
    download_dir = os.path.join(settings.raw_dir, "day=20231101")

    # Create the download folder if it doesn't exist
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # Number of files to download - testing with 10
    num_of_files_to_download = 1000

    # Create a session for making multiple requests
    with requests.Session() as session:

        # Fetch all file names
        file_names = get_file_names(BASE_URL, session)

        # for each file, download the files
        for file_name in file_names[:num_of_files_to_download]:
            file_url = f"{BASE_URL}{file_name}"
            file_path = os.path.join(download_dir, file_name)

            # Download the file using the session
            response = session.get(file_url)
            if response.status_code == 200:
                # Save the file
                with open(file_path, 'wb') as file:
                    file.write(response.content)
            else:
                # Log or handle the error
                print(f"Failed to download file {file_name}")

    return f"Downloaded {num_of_files_to_download} files to {download_dir}"

def get_file_names(base_url: str, session: requests.Session) -> List[str]:
    """Fetches all file names using BeautifulSoup to parse the links
     from the given base URL."""
    # Use the session to make a request
    response = session.get(base_url)

    if response.status_code == 200:

        # Parse HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract links (href attributes) from anchor tags
        links = [a['href'] for a in soup.find_all('a', href=True)]

        # Filter links to keep only those that are relevant to your files
        file_names = [link for link in links if link.endswith('.json.gz')]
        return file_names
    else:
        # Handle the error (e.g., raise an exception or return an empty list)
        print(f"Failed to fetch file names. Status code: {response.status_code}")
        return []
