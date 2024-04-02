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
    s3_access_key_id: str = Field(default="ASIAXB7FCAWSAWVNTJ56", description="Your S3 Access Key ID")
    s3_secret_access_key: str = Field(default="EMe7l8wVn+iAtXt63qQaH8LAQD3gQ93CiyE2y0fs", description="Your S3 Secret Access Key")
    s3_token: str = Field(default="FwoGZXIvYXdzEEYaDDsrsASkq970fpvqgSLAAYci4taNtiHkEsatsEvDe1KeIaLIlj+CoWUEB8SAgkUlcfjNus9vME12XFvocAfm6zMaTpZGM3oe8u0fkn2EI6Ek2zXqO2i638oVGT1xGqLa/4JZPBVGHpX4AKpalffdwzKel+xHrjnF2SAATrA8MYIvMtYQ9Qmto6DbvTz42wWKIurvz+IIKX5ODUOn5g3qc9sbAi3ACZUqXcMG9r1URRuC4xg7wi26HKFgNM6U1T9xYpsbHGCDhqUaw45kjNN2GijI9K+wBjItTUNtjh3Td8qFC1LTOFS7WLnXjOztFSg0cEUXJopoCqORXvb59sF2Hk9ttQY5", description="Your S3 Token")

    model_config = SettingsConfigDict(env_prefix='project_')

    @property
    def raw_dir(self) -> str:
        """Store inside all the raw jsons"""
        return join(self.local_dir, "raw")

    @property
    def prepared_dir(self) -> str:
        return join(self.local_dir, "prepared")