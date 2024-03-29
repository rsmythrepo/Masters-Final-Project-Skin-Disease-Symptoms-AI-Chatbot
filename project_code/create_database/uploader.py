from fastapi import APIRouter, status
from project_code.settings import Settings
import requests
import os

uploader_db = APIRouter(
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Something is wrong with the request"},
    },
    prefix="/api/uploader_db",
    tags=["uploader_db"],
)

@uploader_db.post("/take_fitzpatrick")
def download_file_from_google_drive()-> None:
    fitzpatrick = "https://drive.google.com/uc?id=1B_dHew_vLq3h8QH6ufzA1Eh9N7guB8uj"
    destination_folder = "C:\\Users\\franc\\OneDrive\\Desktop"  # Destination folder where you want to save the file

    # Create the folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    destination = os.path.join(destination_folder, "dataset.zip")
    response = requests.get(fitzpatrick)
    with open(destination, "wb") as file:
        file.write(response.content)

    return "OK"

# @uploader_db.post("/take_DDI")


