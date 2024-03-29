from os.path import dirname, join

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import project_code

PROJECT_DIR = dirname(dirname(project_code.__file__))

class Credentials(BaseSettings):
    """Use env variables prefixed with DB_"""

    host: str
    port: int
    username: str
    password: str
    database: str


    model_config = SettingsConfigDict(env_prefix='db_')


class Settings(BaseSettings):

    local_dir: str = Field(default=join(PROJECT_DIR, "data"),
                           description="For any other value set env variable 'PROJECT_CODE_LOCAL_DIR'")
    telemetry_dsn: str = "http://project2_secret_token@uptrace:14317/2"
    s3_bucket: str = Field(default="YOUR_S3_BUCKET",
                           description="Call the api like `BDI_S3_BUCKET=yourbucket poetry run uvicorn...`")
    s3_access_key_id: str = Field(default="YOUR_ACCESS_KEY_ID", description="Your S3 Access Key ID")
    s3_secret_access_key: str = Field(default="YOUR_SECRET_ACCESS_KEY", description="Your S3 Secret Access Key")
    s3_token: str = Field(default="YOUR_S3_TOKEN", description="Your S3 Token")

    google_drive_token: str = Field(default="YOUR_GOOGLE_DRIVE_TOKEN", description="Your Google Drive Token")

    model_config = SettingsConfigDict(env_prefix='project_')

    @property
    def raw_dir(self) -> str:
        """Store inside all the raw jsons"""
        return join(self.local_dir, "raw")

    @property
    def prepared_dir(self) -> str:
        return join(self.local_dir, "prepared")