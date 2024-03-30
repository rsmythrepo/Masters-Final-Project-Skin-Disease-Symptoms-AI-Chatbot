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
    s3_access_key_id: str = Field(default="ASIAXB7FCAWSA67LV74R", description="Your S3 Access Key ID")
    s3_secret_access_key: str = Field(default="MiHBQCq/JseEfIFrp/be1ACHJOZxFa2wOtzMgnkv", description="Your S3 Secret Access Key")
    s3_token: str = Field(default="FwoGZXIvYXdzEPz//////////wEaDDXE98P/k+Q4E68CDiLAAT4fznPQniGWmbDXFW9YtU4FhztVLaXqkUcN+J8d1+ubHo37VVg/+YotUU/GFlmLw1k7zrgl+9C2g6TGrelMMHYzNjqdNg9j7DnGTt51deSHBuhaxm14PFfjQ+A4zwYUMN2QhseRhkeN0ylGF3SLZaYfnFtSL1m87dG+nJw9bSebDplDH65ldCW7VS88ycnofgNFlmFr1O7TRIb2+KusJdxwWVfBbbgiW8lqagb2tIGeuzyxug6oFvBuzXrVnZdedCjc4Z+wBjItIpxKI/lmj2aUu0ezA0TletFa4EBREJ7wwXVKBcotMKuzkQMj9VdVEYpz+62h", description="Your S3 Token")

    model_config = SettingsConfigDict(env_prefix='project_')

    @property
    def raw_dir(self) -> str:
        """Store inside all the raw jsons"""
        return join(self.local_dir, "raw")

    @property
    def prepared_dir(self) -> str:
        return join(self.local_dir, "prepared")