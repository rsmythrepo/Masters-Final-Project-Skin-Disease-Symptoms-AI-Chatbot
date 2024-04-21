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
    s3_access_key_id: str = Field(default="ASIAXB7FCAWSMLW2T76Q", description="Your S3 Access Key ID")
    s3_secret_access_key: str = Field(default="nHRGiLGsXVjKdDCrJKVLu9V80vqqMnTsipA6Otzf", description="Your S3 Secret Access Key")
    s3_token: str = Field(default="IQoJb3JpZ2luX2VjEEcaCXVzLXdlc3QtMiJIMEYCIQCjdD2GVBzR7a4FRaNRxgpFuSbiDgyWhm6oeMr6K0rKCgIhALjdykqHtcNPcjQy1CMsnDx2eBXdbm55KuIBCc7PnRxyKqkCCHAQABoMNDg1Mjc0ODEzODYwIgwDqbj/1/I+2A1lckEqhgIg1XARc1V6s1nYNDLgt80iSD0rN8VhfiZsUzr71sj4Vmxx2u/P9Dek7fIe/v2EUvf9i1cHUfyQwd/tF3FZgDRw2Htbvyges6ngn0owiqIQXBSaMi1FcOXCuoV6qNADog6dOZ7ktZU76m+M11NNd22KyoCSOZd/hqg6MomWe1B46fNnsNp00mWPIKS8lvJKJ/He+6vmWN/rr5Um3AvIJ7zOGmzJsEAa9kyog8tlPJbZA77eBar0Cv9KBOIktxGFqYSmQzCV7DW9mR6eiJg4AK7yfF24IdOMuxSmKQjis+yvdR1Iq+VU0AH8ueBcA9+EmEg4YHBmWM8/tZMUR1qzQ/uweFlHX4BGMLv4s7AGOpwBdCcHz6OvL/Xxe51rtfloGVUuTyrBnG+jtWl0RBWk4YbBSK8YOz20YuPftXLcMn1bEJMl51+VynS/8YO56LpZC0hBiG6V51JkFejI8QfPv27Si/CEFtw1AhV4FD1Y+5utsc0HKfN46ULEBT5nCLyCiuINHbaiosltGB10sWwVW/S1ZkV6Wbc+FlwKicKtqTfdUpVFEAbzt6pNE36d", description="Your S3 Token")

    model_config = SettingsConfigDict(env_prefix='project_')

    @property
    def raw_dir(self) -> str:
        """Store inside all the raw jsons"""
        return join(self.local_dir, "raw")

    @property
    def prepared_dir(self) -> str:
        return join(self.local_dir, "prepared")