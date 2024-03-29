from os.path import dirname, join

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import project_code

PROJECT_DIR = dirname(dirname(project_code.__file__))

class Credentials(BaseSettings):
    """Use env variables prefixed with DB_"""

    host: str  = "localhost"
    port: int = 5432
    username: str = "postgres"
    password: str = "Luvumbo_98"
    database: str = "postgres"


    model_config = SettingsConfigDict(env_prefix='db_')


class Settings(BaseSettings):

    local_dir: str = Field(default=join(PROJECT_DIR, "data"),
                           description="For any other value set env variable 'PROJECT_CODE_LOCAL_DIR'")
    telemetry_dsn: str = "http://project2_secret_token@uptrace:14317/2"
    s3_bucket: str = Field(default="provafinalproject",
                           description="Call the api like `BDI_S3_BUCKET=yourbucket poetry run uvicorn...`")
    s3_access_key_id: str = Field(default="ASIAXB7FCAWSIJN3XVGV", description="Your S3 Access Key ID")
    s3_secret_access_key: str = Field(default="SRxlGFnspMYcokHAszobJYcu/62j8EtYxLenC7aT", description="Your S3 Secret Access Key")
    s3_token: str = Field(default="FwoGZXIvYXdzEO///////////wEaDHeo9yCqOfYVI1SqYSLAAWRMUp+UVs+ejf4zQOBJXLM8S6g5VZ03DnxuuiuFRmgIyNXwRz6CX69glTQvxWTnWVPEqLr4wjQp2Km7U0xaPA9rykZ7ySaWBSY1bYd111K9tcuVMSaTlVzrYn2tXASqbxnYKFMWkDSTOHmqGGp0ioCRMtxGDsdot0gEId477zkl0p9hRTy2lDCJbauXKwWUlTJhn0H6cxEG8ENlSHbfEKivVZhXUNdMRcBM/DQKLtOdaLrq8IiBhVGYHNTqHIxU8ijH75ywBjItOVFjcFpsCZu8ZwGRytTzEAfm/s0UuPjuWESrDjxu1npwfoftGwDZxvlYBr/i", description="Your S3 Token")

    google_drive_token: str = Field(default="YOUR_GOOGLE_DRIVE_TOKEN", description="Your Google Drive Token")
    google_drive_folder_link: str = Field(default="YOUR_GOOGLE_DRIVE_FOLDER_LINK", description="Your Google Drive Folder Link")
    model_config = SettingsConfigDict(env_prefix='project_')

    @property
    def raw_dir(self) -> str:
        """Store inside all the raw jsons"""
        return join(self.local_dir, "raw")

    @property
    def prepared_dir(self) -> str:
        return join(self.local_dir, "prepared")