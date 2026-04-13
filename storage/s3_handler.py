"""
S3-compatible storage handler for model artifacts, processing logs,
and archival embeddings. Uses LocalStack for local dev, real S3 for prod.
"""
from __future__ import annotations

import json
import io
from datetime import datetime

import boto3
import structlog
from botocore.exceptions import ClientError

from config.settings import settings

log = structlog.get_logger(__name__)


class S3Handler:
    def __init__(self):
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3.endpoint_url,
            aws_access_key_id=settings.s3.access_key_id,
            aws_secret_access_key=settings.s3.secret_access_key,
            region_name=settings.s3.region,
        )
        self._bucket_artifacts = settings.s3.bucket_artifacts
        self._bucket_logs = settings.s3.bucket_logs
        self._ensure_buckets()

    def _ensure_buckets(self):
        for bucket in [self._bucket_artifacts, self._bucket_logs]:
            try:
                self._client.head_bucket(Bucket=bucket)
            except ClientError:
                try:
                    self._client.create_bucket(Bucket=bucket)
                    log.info("s3_bucket_created", bucket=bucket)
                except ClientError as e:
                    log.warning("s3_bucket_create_failed", bucket=bucket, error=str(e))

    def upload_model_artifact(self, key: str, data: bytes) -> str:
        self._client.put_object(
            Bucket=self._bucket_artifacts,
            Key=f"models/{key}",
            Body=data,
        )
        path = f"s3://{self._bucket_artifacts}/models/{key}"
        log.info("model_artifact_uploaded", path=path)
        return path

    def upload_processing_log(self, camera_id: str, log_data: dict) -> str:
        ts = datetime.utcnow().strftime("%Y/%m/%d/%H")
        key = f"logs/{camera_id}/{ts}/{datetime.utcnow().isoformat()}.json"
        self._client.put_object(
            Bucket=self._bucket_logs,
            Key=key,
            Body=json.dumps(log_data).encode(),
            ContentType="application/json",
        )
        path = f"s3://{self._bucket_logs}/{key}"
        log.debug("processing_log_uploaded", path=path)
        return path

    def upload_embeddings_batch(self, camera_id: str, embeddings: list[dict]) -> str:
        ts = datetime.utcnow().strftime("%Y/%m/%d/%H")
        key = f"embeddings/{camera_id}/{ts}/batch_{datetime.utcnow().timestamp():.0f}.jsonl"
        body = "\n".join(json.dumps(e) for e in embeddings)
        self._client.put_object(
            Bucket=self._bucket_logs,
            Key=key,
            Body=body.encode(),
            ContentType="application/x-ndjson",
        )
        path = f"s3://{self._bucket_logs}/{key}"
        log.debug("embeddings_batch_uploaded", path=path, count=len(embeddings))
        return path
