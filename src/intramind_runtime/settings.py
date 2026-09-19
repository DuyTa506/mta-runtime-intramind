from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .artifacts import MAX_ARTIFACT_BYTES


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RUNTIME_", extra="ignore")
    database_url: SecretStr
    service_token: SecretStr = Field(min_length=32)
    minio_endpoint: str
    minio_access_key: SecretStr
    minio_secret_key: SecretStr
    minio_bucket: str
    minio_secure: bool = True
    artifact_max_bytes: int = Field(default=MAX_ARTIFACT_BYTES, gt=0, le=MAX_ARTIFACT_BYTES)
    artifact_upload_concurrency: int = Field(default=2, ge=1, le=8)
    temporal_address: str = "temporal:7233"
    temporal_namespace: str = "intramind"
    temporal_queue: str = "intramind-control"
    lease_seconds: int = Field(default=60, ge=10)
    executor_count: int = Field(default=1, ge=1, le=128)
    pool_config: str = "config/pools.json"
    log_level: str = "INFO"
