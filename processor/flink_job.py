"""
PyFlink streaming job: consumes raw video frames from Kafka, performs
metadata extraction, checks the Redis embedding cache, runs inference
on cache misses, and emits enriched results.

Performance target: sub-50ms per-frame processing window.

Tuning knobs:
  - parallelism.default          → number of parallel operators
  - checkpoint interval          → fault-tolerance vs latency tradeoff
  - buffer timeout               → network stack flushing frequency
  - taskmanager.numberOfTaskSlots → concurrency per TaskManager
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time

import numpy as np
import structlog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import settings
from processor.frame_processor import decode_frame_message, FrameMetadata
from processor.cache_handler import EmbeddingCache
from inference.model_handler import ModelHandler
from monitoring.metrics_exporter import MetricsCollector, start_metrics_server
from storage.s3_handler import S3Handler

log = structlog.get_logger(__name__)


class VoyagerStreamProcessor:
    """Standalone stream processor that mirrors what the PyFlink MapFunction does.

    Designed to run either inside Flink (via a map UDF wrapper) or as a
    standalone consumer for local development and testing.
    """

    EMBEDDINGS_BATCH_SIZE = 50

    def __init__(self):
        self.cache = EmbeddingCache()
        self.model = ModelHandler()
        self.metrics = MetricsCollector()
        self._processed = 0
        self._embedding_buffers: dict[str, list[dict]] = {}

        # S3/LocalStack is optional local infra; any failure to reach it
        # (unlike Redis, boto3 doesn't degrade gracefully on its own) should
        # disable archival rather than take down frame processing.
        try:
            self.storage = S3Handler()
        except Exception as e:
            log.warning("s3_handler_unavailable", error=str(e))
            self.storage = None

    def process_frame(self, raw_value: bytes) -> dict:
        """Full processing pipeline for a single frame message.

        Returns an enriched result dict with metadata, embedding info,
        cache status, and latency breakdown.
        """
        t_start = time.perf_counter()

        meta, frame = decode_frame_message(raw_value)
        t_decode = time.perf_counter()

        cache_result = self.cache.lookup(meta.content_hash)
        t_cache = time.perf_counter()

        if cache_result.hit:
            embedding = cache_result.embedding
            inference_ms = 0.0
            cache_status = "HIT"
        else:
            embedding = self.model.generate_embedding(frame)
            t_infer = time.perf_counter()
            inference_ms = (t_infer - t_cache) * 1000
            self.cache.store(meta.content_hash, embedding)
            cache_status = "MISS"

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        result = {
            **meta.to_dict(),
            "embedding_dim": len(embedding),
            "cache_status": cache_status,
            "latency": {
                "decode_ms": round((t_decode - t_start) * 1000, 3),
                "cache_lookup_ms": round(cache_result.lookup_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "total_ms": round(total_ms, 3),
            },
            "cache_stats": self.cache.get_stats(),
        }

        self.metrics.record_frame_processed(
            camera_id=meta.camera_id,
            total_ms=total_ms,
            cache_hit=cache_result.hit,
            inference_ms=inference_ms,
        )

        if self.storage is not None:
            self._buffer_for_archival(meta.camera_id, meta.content_hash, cache_status, embedding)

        self._processed += 1
        if self._processed % 50 == 0:
            log.info(
                "processing_checkpoint",
                frames_processed=self._processed,
                avg_latency_ms=round(total_ms, 2),
                cache_hit_rate=self.cache.hit_rate,
            )

        return result

    def _buffer_for_archival(
        self, camera_id: str, content_hash: str, cache_status: str, embedding: np.ndarray
    ) -> None:
        """Accumulate embeddings per camera and flush to S3 once a batch fills.

        Archival is best-effort: an upload failure is logged and the batch is
        dropped rather than retried, so a flaky S3/LocalStack never blocks or
        crashes frame processing.
        """
        buffer = self._embedding_buffers.setdefault(camera_id, [])
        buffer.append({
            "content_hash": content_hash,
            "cache_status": cache_status,
            "embedding": embedding.tolist(),
        })

        if len(buffer) >= self.EMBEDDINGS_BATCH_SIZE:
            try:
                self.storage.upload_embeddings_batch(camera_id, buffer)
            except Exception as e:
                log.warning("s3_upload_failed", camera_id=camera_id, error=str(e))
            finally:
                self._embedding_buffers[camera_id] = []


def _find_kafka_connector_jar() -> str:
    """Locate the Flink Kafka connector JAR in the lib/ directory."""
    import glob
    project_root = os.path.join(os.path.dirname(__file__), "..")
    patterns = [
        os.path.join(project_root, "lib", "flink-sql-connector-kafka-*.jar"),
        os.path.join(project_root, "lib", "flink-connector-kafka-*.jar"),
    ]
    for pattern in patterns:
        jars = glob.glob(pattern)
        if jars:
            return os.path.abspath(jars[0])
    raise FileNotFoundError(
        "Kafka connector JAR not found in lib/. "
        "Download it: curl -L -o lib/flink-sql-connector-kafka-4.0.1-2.0.jar "
        "https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/4.0.1-2.0/flink-sql-connector-kafka-4.0.1-2.0.jar"
    )


def create_flink_pipeline():
    """Build and return a PyFlink streaming pipeline.

    This wires up:
      1. Kafka source (deserialization of raw frame messages)
      2. Map operator (VoyagerStreamProcessor.process_frame)
      3. Kafka sink (enriched results)

    Checkpointing and parallelism are configured for sub-50ms latency.
    """
    try:
        from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
        from pyflink.datastream.connectors.kafka import (
            KafkaSource, KafkaOffsetsInitializer, KafkaSink,
            KafkaRecordSerializationSchema,
        )
        from pyflink.common.serialization import SimpleStringSchema
        from pyflink.common import WatermarkStrategy, Configuration, Types
        from pyflink.datastream.functions import MapFunction, RuntimeContext
    except ImportError:
        log.error("pyflink_not_available", msg="Install apache-flink to run the Flink pipeline")
        raise

    config = Configuration()
    config.set_string("rest.port", "8082")
    config.set_string("rest.bind-address", "0.0.0.0")
    config.set_string("web.submit.enable", "true")

    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)

    kafka_jar = _find_kafka_connector_jar()
    env.add_jars(f"file://{kafka_jar}")
    log.info("kafka_connector_loaded", jar=kafka_jar)

    # ── Sub-50ms tuning ──────────────────────────────────────────
    env.set_parallelism(settings.flink.parallelism)
    env.enable_checkpointing(settings.flink.checkpoint_interval_ms)

    env.set_buffer_timeout(1)  # 1ms buffer timeout

    kafka_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(settings.kafka.bootstrap_servers)
        .set_topics(settings.kafka.topic_frames)
        .set_group_id(settings.kafka.group_id)
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .set_property("session.timeout.ms", "30000")
        .set_property("heartbeat.interval.ms", "5000")
        .set_property("max.poll.records", "100")
        .set_property("fetch.max.wait.ms", "200")
        .set_property("partition.discovery.interval.ms", "30000")
        .build()
    )

    class FrameMapFunction(MapFunction):
        def __init__(self):
            self.processor = None
            self._errors = 0

        def open(self, runtime_context: RuntimeContext):
            self.processor = VoyagerStreamProcessor()
            subtask = runtime_context.get_index_of_this_subtask()
            try:
                start_metrics_server(port=8000 + subtask)
            except OSError:
                log.warning("metrics_port_in_use", subtask=subtask)
            log.info("flink_map_function_opened", subtask=subtask,
                     metrics_port=8000 + subtask)

        def map(self, value: str) -> str:
            try:
                result = self.processor.process_frame(value.encode("utf-8"))
                return json.dumps(result)
            except Exception as e:
                self._errors += 1
                log.error("frame_processing_error",
                          error=str(e), error_count=self._errors)
                return json.dumps({
                    "error": str(e),
                    "error_count": self._errors,
                })

    stream = env.from_source(
        kafka_source,
        WatermarkStrategy.no_watermarks(),
        "kafka-frame-source",
    )

    processed = stream.map(FrameMapFunction(), output_type=Types.STRING())

    serializer = (
        KafkaRecordSerializationSchema.builder()
        .set_topic(settings.kafka.topic_processed)
        .set_value_serialization_schema(SimpleStringSchema())
        .build()
    )

    sink = (
        KafkaSink.builder()
        .set_bootstrap_servers(settings.kafka.bootstrap_servers)
        .set_record_serializer(serializer)
        .set_property("transaction.timeout.ms", "600000")
        .build()
    )

    processed.sink_to(sink)

    return env


def run_standalone_consumer(max_messages: int | None = None):
    """Run the processor as a standalone Kafka consumer (no Flink dependency).

    Useful for local development, debugging, and benchmarking.
    """
    from confluent_kafka import Consumer

    start_metrics_server(port=8000)

    consumer = Consumer({
        "bootstrap.servers": settings.kafka.bootstrap_servers,
        "group.id": settings.kafka.group_id,
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
        "max.poll.interval.ms": 300000,
    })
    consumer.subscribe([settings.kafka.topic_frames])

    processor = VoyagerStreamProcessor()
    count = 0

    log.info("standalone_consumer_started", topic=settings.kafka.topic_frames)

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                log.error("consumer_error", error=msg.error())
                continue

            result = processor.process_frame(msg.value())
            count += 1

            if count % 10 == 0:
                log.info(
                    "consumer_progress",
                    processed=count,
                    last_latency_ms=result["latency"]["total_ms"],
                    cache_hit_rate=processor.cache.hit_rate,
                )

            if max_messages and count >= max_messages:
                break
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        log.info("standalone_consumer_stopped", total_processed=count)

    return processor.metrics.get_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voyager Stream Processor")
    parser.add_argument("--mode", choices=["flink", "standalone"], default="standalone")
    parser.add_argument("--max-messages", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "flink":
        env = create_flink_pipeline()
        env.execute("voyager-frame-processing")
    else:
        run_standalone_consumer(max_messages=args.max_messages)
