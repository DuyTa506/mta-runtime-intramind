from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RUNTIME_", extra="ignore")
    database_url: SecretStr
    service_token: SecretStr = Field(min_length=32)
    minio_endpoint: str
    minio_access_key: SecretStr
    minio_secret_key: SecretStr
    minio_bucket: str
    minio_secure: bool = True
    temporal_address: str = "temporal:7233"
    temporal_namespace: str = "intramind"
    temporal_queue: str = "intramind-control"
    lease_seconds: int = Field(default=60, ge=10)
    executor_count: int = Field(default=1, ge=1, le=128)
    pool_config: str = "config/pools.json"
    log_level: str = "INFO"
