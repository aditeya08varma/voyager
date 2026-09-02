"""Tests for the S3/LocalStack storage handler."""
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from storage.s3_handler import S3Handler


def _client_error():
    return ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket")


@pytest.fixture
def mock_client():
    with patch("storage.s3_handler.boto3.client") as make_client:
        client = MagicMock()
        make_client.return_value = client
        yield client


class TestBucketBootstrap:
    def test_existing_buckets_are_not_recreated(self, mock_client):
        mock_client.head_bucket.return_value = {}
        S3Handler()
        mock_client.create_bucket.assert_not_called()

    def test_missing_buckets_are_created(self, mock_client):
        mock_client.head_bucket.side_effect = _client_error()
        S3Handler()
        assert mock_client.create_bucket.call_count == 2  # artifacts + logs


class TestUploadModelArtifact:
    def test_uploads_to_artifacts_bucket_under_models_prefix(self, mock_client):
        mock_client.head_bucket.return_value = {}
        handler = S3Handler()

        path = handler.upload_model_artifact("mobilenetv2/v1.pt", b"weights")

        mock_client.put_object.assert_called_once_with(
            Bucket=handler._bucket_artifacts, Key="models/mobilenetv2/v1.pt", Body=b"weights"
        )
        assert path == f"s3://{handler._bucket_artifacts}/models/mobilenetv2/v1.pt"


class TestUploadProcessingLog:
    def test_uploads_json_to_logs_bucket(self, mock_client):
        mock_client.head_bucket.return_value = {}
        handler = S3Handler()

        path = handler.upload_processing_log("CAM-001", {"frames": 5})

        assert mock_client.put_object.call_count == 1
        kwargs = mock_client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == handler._bucket_logs
        assert kwargs["Key"].startswith("logs/CAM-001/")
        assert kwargs["ContentType"] == "application/json"
        assert path.startswith(f"s3://{handler._bucket_logs}/logs/CAM-001/")


class TestUploadEmbeddingsBatch:
    def test_uploads_ndjson_with_one_line_per_embedding(self, mock_client):
        mock_client.head_bucket.return_value = {}
        handler = S3Handler()

        embeddings = [{"hash": "a"}, {"hash": "b"}]
        path = handler.upload_embeddings_batch("CAM-001", embeddings)

        kwargs = mock_client.put_object.call_args.kwargs
        assert kwargs["ContentType"] == "application/x-ndjson"
        body_lines = kwargs["Body"].decode().splitlines()
        assert len(body_lines) == 2
        assert path.startswith(f"s3://{handler._bucket_logs}/embeddings/CAM-001/")
