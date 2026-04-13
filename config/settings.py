from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class KafkaSettings(BaseSettings):
    bootstrap_servers: str = Field("localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    topic_frames: str = Field("voyager.frames.raw", alias="KAFKA_TOPIC_FRAMES")
    topic_processed: str = Field("voyager.frames.processed", alias="KAFKA_TOPIC_PROCESSED")
    topic_metrics: str = Field("voyager.metrics", alias="KAFKA_TOPIC_METRICS")
    group_id: str = Field("voyager-processor", alias="KAFKA_GROUP_ID")
    message_max_bytes: int = 10_485_760

    model_config = {"env_file": ".env", "extra": "ignore"}


class RedisSettings(BaseSettings):
    host: str = Field("localhost", alias="REDIS_HOST")
    port: int = Field(6379, alias="REDIS_PORT")
    db: int = Field(0, alias="REDIS_DB")
    embedding_ttl: int = Field(3600, alias="REDIS_EMBEDDING_TTL")
    max_connections: int = Field(20, alias="REDIS_MAX_CONNECTIONS")

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class S3Settings(BaseSettings):
    endpoint_url: str = Field("http://localhost:4566", alias="AWS_ENDPOINT_URL")
    access_key_id: str = Field("test", alias="AWS_ACCESS_KEY_ID")
    secret_access_key: str = Field("test", alias="AWS_SECRET_ACCESS_KEY")
    region: str = Field("us-east-1", alias="AWS_DEFAULT_REGION")
    bucket_artifacts: str = Field("voyager-artifacts", alias="S3_BUCKET_ARTIFACTS")
    bucket_logs: str = Field("voyager-logs", alias="S3_BUCKET_LOGS")

    model_config = {"env_file": ".env", "extra": "ignore"}


class FlinkSettings(BaseSettings):
    jobmanager_host: str = Field("localhost", alias="FLINK_JOBMANAGER_HOST")
    jobmanager_port: int = Field(8081, alias="FLINK_JOBMANAGER_PORT")
    parallelism: int = Field(2, alias="FLINK_PARALLELISM")
    checkpoint_interval_ms: int = Field(5000, alias="FLINK_CHECKPOINT_INTERVAL_MS")

    model_config = {"env_file": ".env", "extra": "ignore"}


class ModelSettings(BaseSettings):
    name: str = Field("mobilenetv2", alias="MODEL_NAME")
    cache_dir: str = Field("./data/model_cache", alias="MODEL_CACHE_DIR")
    embedding_dim: int = Field(1280, alias="EMBEDDING_DIM")

    model_config = {"env_file": ".env", "extra": "ignore"}


class ProducerSettings(BaseSettings):
    frame_interval_ms: int = Field(100, alias="FRAME_INTERVAL_MS")
    video_source: str = Field("./data/sample_videos/sample.mp4", alias="VIDEO_SOURCE")
    num_cameras: int = Field(4, alias="NUM_SIMULATED_CAMERAS")

    model_config = {"env_file": ".env", "extra": "ignore"}


class VoyagerSettings(BaseSettings):
    kafka: KafkaSettings = KafkaSettings()
    redis: RedisSettings = RedisSettings()
    s3: S3Settings = S3Settings()
    flink: FlinkSettings = FlinkSettings()
    model: ModelSettings = ModelSettings()
    producer: ProducerSettings = ProducerSettings()

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = VoyagerSettings()
