from os.path import dirname, join

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import project_code

PROJECT_DIR = dirname(dirname(project_code.__file__))


class Settings(BaseSettings):

    local_dir: str = Field(default=join(PROJECT_DIR, "data"),
                           description="For any other value set env variable 'PROJECT_CODE_LOCAL_DIR'")
    telemetry_dsn: str = "http://project2_secret_token@uptrace:14317/2"
    s3_bucket: str = Field(default="bucket name",
                           description="Call the api like `BDI_S3_BUCKET=yourbucket poetry run uvicorn...`")
    model_config = SettingsConfigDict(env_prefix='project_')

    @property
    def raw_dir(self) -> str:
        """Store inside all the raw jsons"""
        return join(self.local_dir, "raw")

    @property
    def prepared_dir(self) -> str:
        return join(self.local_dir, "prepared")