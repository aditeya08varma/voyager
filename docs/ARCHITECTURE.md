# Voyager — Architecture & Technical Deep Dive

*A source-verified walkthrough of how Voyager actually works: every claim below is
traced to a specific file, function, or config value in this repo. Where the
system's own documentation (`README.md`, `loadtest/scale_design.md`) describes a
target or an aspiration rather than what's running today, that distinction is
called out explicitly rather than blurred.*

---

## What Voyager is

Voyager is a local, fully-runnable simulation of a real-time video ingestion
pipeline: simulated camera feeds publish frames to Kafka, a stream processor
(either PyFlink or a plain Kafka consumer) decodes each frame, checks a Redis
cache before paying for AI inference, runs a MobileNetV2 model on cache misses,
and exposes Prometheus metrics that a bundled Grafana dashboard visualizes.
Every piece of infrastructure it talks to — Kafka, Flink, Redis, S3 (via
LocalStack), Prometheus, Grafana — is defined in [`docker-compose.yml`](../docker-compose.yml)
and runs on a laptop.

## Why it's built this way

The repository documents its own intent directly. [`loadtest/scale_design.md`](../loadtest/scale_design.md)
closes with a section titled **"Interview Talking Points"** — five rehearsed
answers to questions like *"How do you handle 10TB/day?"* and *"How do you keep
latency under 50ms?"*. That framing is the honest one to use here: Voyager is a
systems-design exercise built around the specific mechanics that come up in
distributed-systems interviews — partition-key ordering, exactly-once
checkpointing, cache-driven cost reduction, and back-of-envelope capacity math —
implemented as a real, runnable pipeline rather than left as a whiteboard
sketch. The "Known Gaps" section at the end of this document is written in that
same spirit: precise about what's real versus what's still a design note.

---

## Architecture at a glance

```mermaid
flowchart LR
    subgraph Cameras["producer/ — simulated camera feeds"]
        C1[CameraSimulator thread]
        C2[CameraSimulator thread]
        C3[CameraSimulator thread]
        C4[CameraSimulator thread]
    end

    C1 & C2 & C3 & C4 -->|"JPEG q80 → base64 JSON\nkey = camera_id"| K[("Kafka topic\nvoyager.frames.raw\n6 partitions · LZ4")]

    K --> P{{"processor.flink_job\n--mode flink | --mode standalone"}}
    P -->|"embedded PyFlink\nMiniCluster"| F["VoyagerStreamProcessor\n.process_frame()"]
    P -->|"confluent_kafka\nConsumer"| F

    F --> D["frame_processor\n.decode_frame_message()\ndecode JPEG → grayscale →\ncontent_hash + edge_density"]
    D --> L{"EmbeddingCache.lookup()\nRedis GET voyager:emb:{hash}"}
    L -->|HIT| E1["cached 1280-d embedding\n(< 1ms)"]
    L -->|MISS| M["ModelHandler\n.generate_embedding()\nMobileNetV2, CPU"]
    M --> S["EmbeddingCache.store()\nRedis SETEX, TTL 3600s"]
    S --> E1

    E1 --> R["enriched JSON result\n(metadata + embedding_dim + latency)"]
    R -->|"Flink mode only"| KP[("Kafka topic\nvoyager.frames.processed")]

    F -.->|"metrics recorded\nper frame"| PM["MetricsCollector\n(monitoring/)"]
    PM --> PR["Prometheus\n:8000-8005"]
    PR --> GR["Grafana :3000\n14-panel dashboard"]

    S3H["storage.S3Handler\n(fully implemented)"]
    R -.->|"no call site anywhere\nin the live pipeline"| S3H
```

The dashed line into `S3Handler` is deliberate — see [Known Gaps, #2](#2-s3handler-is-fully-built-and-never-called).

---

## The frame lifecycle, precisely

Everything downstream of Kafka funnels through one method:
[`VoyagerStreamProcessor.process_frame`](../processor/flink_job.py) (`processor/flink_job.py:49-106`).
It's the same code path whether it's wrapped in a Flink `MapFunction` or called
directly by the standalone consumer loop — that's the whole point of the class:
one implementation, two runtimes.

```mermaid
sequenceDiagram
    participant Kafka as Kafka (voyager.frames.raw)
    participant VSP as VoyagerStreamProcessor
    participant FP as decode_frame_message()
    participant Cache as EmbeddingCache (Redis)
    participant Model as ModelHandler (MobileNetV2)
    participant Metrics as MetricsCollector

    Kafka->>VSP: raw_value (bytes)
    VSP->>FP: decode_frame_message(raw_value)
    FP-->>VSP: (FrameMetadata, frame ndarray)
    VSP->>Cache: lookup(content_hash)
    alt cache hit
        Cache-->>VSP: CacheResult(hit=True, embedding)
    else cache miss
        Cache-->>VSP: CacheResult(hit=False)
        VSP->>Model: generate_embedding(frame)
        Model-->>VSP: 1280-d embedding
        VSP->>Cache: store(content_hash, embedding)
    end
    VSP->>Metrics: record_frame_processed(camera_id, total_ms, cache_hit, inference_ms)
    VSP-->>Kafka: enriched result JSON (sunk to voyager.frames.processed in Flink mode)
```

Four `time.perf_counter()` checkpoints inside `process_frame` produce the
`latency.decode_ms`, `latency.cache_lookup_ms`, `latency.inference_ms`, and
`latency.total_ms` fields that land in every result and in the Grafana
"Latency & Performance" row.

---

## Component deep dive

### Producer layer (`producer/`)

`kafka_producer.py` defines `create_producer()`, tuned for throughput rather
than per-message latency:

| Setting | Value | Effect |
|---|---|---|
| `linger.ms` | 5 | batches messages for up to 5ms before sending |
| `batch.num.messages` | 100 | target batch size |
| `compression.type` | `lz4` | matches the README's "LZ4 compression" claim |
| `acks` | `1` | leader ack only — no replication wait (there's only 1 broker in dev) |
| `queue.buffering.max.messages` | 100,000 | producer-side buffer cap |

Each `CameraSimulator` (`producer/kafka_producer.py:68-147`) is a daemon thread
that opens a video with OpenCV, loops it (`CAP_PROP_POS_FRAMES, 0` on EOF),
JPEG-encodes each frame at quality 80, base64-encodes it into a `FrameMessage`,
and publishes with **`key=camera_id`** — this is what guarantees per-camera
ordering within a partition, since Kafka only orders messages that share a key.

There are **two separate entry points** that both spin up camera simulators,
and they disagree with each other:

- `kafka_producer.run_multi_camera_producer()` — reads `settings.producer.num_cameras`
  (default 4) and `settings.producer.video_source` from `.env`/`config/settings.py`,
  auto-generating a sample video if the source is missing. This is the one the
  README's Quick Start points at.
- `multi_camera_producer.py` — a second, independent script whose docstring
  says *"Spawns 4 concurrent camera simulators"* but whose code hardcodes a
  list of **6** files (`virat_test_01.mp4` … `virat_test_06.mp4`) at a fixed
  `frame_interval_ms=33` (~30 FPS), ignoring `config/settings.py` entirely.

`sample_video_generator.py` builds a synthetic 30-second MP4 specifically to
exercise the cache: every 20-frame window, frames `0-2` are drawn from a pool
of 5 pre-rendered static backgrounds (`_generate_backgrounds`) while the rest
are procedurally animated (`_render_dynamic_frame`). The recurring backgrounds
are what actually produce cache hits when this video is played through the
pipeline — see the caching section below for why that distinction matters.

### Frame decoding & metadata (`processor/frame_processor.py`)

`decode_frame_message()` (`processor/frame_processor.py:52-86`) parses the Kafka
JSON payload with `orjson`, base64-decodes the JPEG, and runs `cv2.imdecode`.
From the decoded BGR frame it derives:

- **`content_hash`** — a 64-bit difference hash, computed by resizing the
  grayscale frame to 9×8, comparing each pixel to its right-hand neighbor, and
  packing the 64 boolean results into a hex string (`_compute_content_hash`,
  lines 89-98). This is the cache key.
- **`mean_brightness`** — `np.mean` of the grayscale frame.
- **`edge_density`** — fraction of pixels flagged by `cv2.Canny(gray, 50, 150)`,
  used as a scene-activity signal (not currently consumed downstream — it rides
  along in `FrameMetadata` but nothing branches on it today).

```mermaid
flowchart LR
    A[BGR frame] --> B["cvtColor → grayscale"]
    B --> C["cv2.resize to 9×8\n(INTER_AREA)"]
    C --> D["diff[x,y] =\npixel[x,y+1] > pixel[x,y]"]
    D --> E["pack 64 bits\ninto an integer"]
    E --> F["format as 16-char hex\n= content_hash"]
    F --> G["Redis key\nvoyager:emb:{content_hash}"]
```

### The cache layer — what "perceptual hash caching" actually means here

This is worth being precise about, because the README's phrasing ("perceptual
hash caching") suggests fuzzy, near-duplicate matching, and that's **not**
what's implemented.

`EmbeddingCache` (`processor/cache_handler.py`) does a plain Redis `GET`/`SETEX`
keyed by the **exact** `content_hash` string:

```python
def lookup(self, content_hash: str) -> CacheResult:
    key = f"{self.PREFIX}{content_hash}"
    raw = self._r.get(key)          # exact string match, no distance check
    ...
```

Two frames only collide in the cache if their 9×8-resized difference pattern is
**bit-for-bit identical**. That's exactly right for the synthetic video's
recurring static backgrounds (which really are pixel-identical on repeat), and
it's a legitimate, cheap way to skip inference on truly static scenes — but it
is an *exact-match* cache, not a *similarity-threshold* one.

Separately, `inference/perceptual_hash.py` implements a genuine similarity API:
`dhash()`, `ahash()`, `phash()`, `hamming_distance()`, and `are_similar(threshold=10)`.
It is fully written and unit-testable — and, confirmed by a repo-wide grep, it
is **never imported or called anywhere** in `producer/`, `processor/`,
`inference/`, `loadtest/`, or `tests/`. It's dead code today. Wiring it into
`EmbeddingCache.lookup()` as a fallback (scan a small set of recent hashes for
one within Hamming distance ≤ N before falling back to inference) is the single
change that would make the "perceptual hash caching" description literally
true. See [Known Gaps, #1](#1-perceptual_hashpy-is-fully-built-and-never-used).

### AI inference (`inference/model_handler.py`)

`ModelHandler` lazily loads `torchvision.models.mobilenet_v2` with
`MobileNet_V2_Weights.DEFAULT`, replaces the classifier head with
`torch.nn.Identity()` to expose the 1280-dim feature vector directly, and runs
the standard ImageNet preprocessing (`Resize(256)` → `CenterCrop(224)` →
`Normalize` with ImageNet mean/std). `_select_device()` picks CUDA, then MPS,
then falls back to CPU.

### Metrics (`monitoring/metrics_exporter.py`)

`MetricsCollector` is a thread-safe singleton (`__new__` + a class-level
`threading.Lock`) exposing seven Prometheus series: `voyager_frames_processed_total`,
`voyager_frame_processing_seconds`, `voyager_cache_lookups_total{status}`,
`voyager_inference_duration_seconds`, `voyager_cache_hit_rate`,
`voyager_inferences_skipped_total`, and `voyager_throughput_fps`. Throughput is
computed from a rolling 5-second window of timestamps, rebuilt with a list
comprehension on every single call to `record_frame_processed` — see
[Known Gaps, #10](#10-the-throughput-gauge-recomputes-a-list-on-every-frame).

`start_metrics_server(port)` is called once per Flink subtask, on
`8000 + subtask_index` (`processor/flink_job.py:193`) — which is exactly why
`monitoring/prometheus.yml` scrapes six static targets, `:8000` through `:8005`,
anticipating up to six parallel subtasks even though `FLINK_PARALLELISM`
defaults to 2.

Throughput is computed from a rolling 5-second window of timestamps held in a
`collections.deque`, with expired entries popped from the front on each call —
O(1) amortized per frame (see Known Gaps, #10, for the list-based version this
replaced).

### Storage (`storage/s3_handler.py`)

`S3Handler` is complete: it auto-creates its two buckets against the LocalStack
endpoint on init, and exposes `upload_model_artifact`, `upload_processing_log`,
and `upload_embeddings_batch` (JSON-lines). None of these three methods has a
call site outside `storage/s3_handler.py` itself. The architecture diagram's
"→ S3" arrow in the README describes a capability that exists but isn't wired
in yet.

### Orchestration modes (`processor/flink_job.py`)

`python -m processor.flink_job --mode flink` calls `create_flink_pipeline()`,
which builds a real `KafkaSource → map(FrameMapFunction) → KafkaSink` topology,
configures parallelism and checkpointing from `settings.flink`, sets a 1ms
buffer timeout for low latency, and requires the Kafka connector JAR at
`lib/flink-sql-connector-kafka-4.0.1-2.0.jar` (already committed to the repo).
Critically, it calls `StreamExecutionEnvironment.get_execution_environment(config)`
with no remote target — in PyFlink that spins up an **embedded local
MiniCluster inside the Python process**, not a submission to a remote cluster.
That's why the code hardcodes `rest.port = "8082"` — it's giving *that*
embedded MiniCluster's own web UI a port that doesn't collide with the
docker-compose Flink cluster's UI on `8081`. See
[Known Gaps, #3](#3-the-docker-compose-flink-cluster-is-never-actually-used).

`--mode standalone` calls `run_standalone_consumer()`, a plain
`confluent_kafka.Consumer` loop with no Flink dependency at all — useful for
local development without the JVM, and functionally equivalent per-frame since
both modes call the same `VoyagerStreamProcessor.process_frame`.

---

## Local infrastructure topology

```mermaid
flowchart TB
    subgraph Compose["docker-compose.yml"]
        ZK["zookeeper : 2181"]
        KFK["kafka : 9092 / 29092\n6 partitions · 2h retention"]
        FJM["flink-jobmanager : 8081"]
        FTM["flink-taskmanager"]
        RD["redis : 6379\n512MB · allkeys-lru · AOF"]
        LS["localstack : 4566\n(S3 emulation)"]
        PROM["prometheus : 9090"]
        GRAF["grafana : 3000\nadmin / voyager"]
        RDEXP["redis-exporter : 9121"]
    end

    ZK --> KFK
    FJM --> FTM
    RD --> RDEXP
    RDEXP --> PROM
    FTM -->|"metrics :9249"| PROM
    PROM --> GRAF

    APP["python -m processor.flink_job --mode flink\n(embedded PyFlink MiniCluster, UI on :8082)"]
    APP -.->|"not currently submitted to"| FJM
    APP --> KFK
    APP --> RD
```

The Grafana dashboard (`monitoring/grafana/dashboard.json`) ships 14 panels
across 4 rows:

| Row | Panels |
|---|---|
| Pipeline Overview | Total Frames Processed, Cache Hit Rate, Inferences Skipped, Throughput (FPS) |
| Latency & Performance | Processing Latency P50/P95/P99, AI Inference Duration |
| Cache Performance | Cache Hit vs Miss Rate, Redis Memory Usage |
| Per-Camera Breakdown | Frames Processed by Camera, Latency by Camera |

---

## Configuration surface

All settings live in `config/settings.py` as Pydantic `BaseSettings`, loaded
from `.env` (gitignored — the repo correctly never commits it).

| Group | Key env vars | Notes |
|---|---|---|
| Kafka | `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC_FRAMES`, `KAFKA_TOPIC_PROCESSED`, `KAFKA_GROUP_ID` | |
| Redis | `REDIS_HOST`, `REDIS_PORT`, `REDIS_EMBEDDING_TTL` (default 3600s) | |
| S3 | `AWS_ENDPOINT_URL`, `S3_BUCKET_ARTIFACTS`, `S3_BUCKET_LOGS` | only read by `S3Handler`, which nothing else calls |
| Flink | `FLINK_PARALLELISM` (default 2), `FLINK_CHECKPOINT_INTERVAL_MS` | |
| Model | `MODEL_NAME`, `EMBEDDING_DIM` (1280) | |
| Producer | `FRAME_INTERVAL_MS`, `VIDEO_SOURCE`, `NUM_SIMULATED_CAMERAS` | ignored by `multi_camera_producer.py` |

`.env` also defines `PROMETHEUS_PORT`, `GRAFANA_PORT`, and `METRICS_EXPORT_PORT` —
none of these three has a matching `Field` in `config/settings.py` or is read
anywhere in the codebase; the real ports are hardcoded in `docker-compose.yml`
and `flink_job.py`. (`KAFKA_TOPIC_METRICS` was the fourth dead key in this
category — it has since been removed from `KafkaSettings`.)

---

## Testing & verification

| File | Covers |
|---|---|
| `tests/test_processor.py` | `decode_frame_message`, `_compute_content_hash` determinism, `_compute_edge_density` |
| `tests/test_producer.py` | `FrameMessage` construction, `generate_sample_video` output, `CameraSimulator.stop()` |
| `tests/test_load.py` | `_generate_test_frame` determinism, encode/decode roundtrip, `LoadTestResult` construction |

**Not covered by any test today:** `cache_handler.py` (`EmbeddingCache`),
`model_handler.py`, `s3_handler.py`, `metrics_exporter.py`, and `flink_job.py`
(`VoyagerStreamProcessor`, `create_flink_pipeline`). A stale
`tests/__pycache__/test_cache.cpython-312-pytest-9.0.3.pyc` is the only trace
of a `test_cache.py` that once existed and was removed without a replacement —
worth restoring first, since `EmbeddingCache` is the most-connected module in
the codebase.

`loadtest/stress_test.py` is the only place the cache's cost-saving effect is
actually exercised, via `test_cache_hit_simulation()`. Its default parameters
(50 unique frames, then 150 repeats drawn from those same 50) mean every one of
the 150 repeats is guaranteed to hit — the math works out to a **75% hit rate**,
not the README's headline **40%+**. That 40% figure traces to
`loadtest/scale_design.md`'s production-scale assumption, not to a measurement
taken against this repo. See [Known Gaps, #6](#6-the-only-measured-cache-hit-rate-is-75-not-40).

---

## The 10TB/day scaling design — what's real vs. what's a plan

`loadtest/scale_design.md` is an explicit, well-reasoned target architecture:
a 5-broker/3-AZ Kafka cluster with 120 partitions, an 8-node Flink cluster with
RocksDB state and S3 checkpoint storage, a 6-master Redis Cluster with LFU
eviction and cache warming, tiered S3 storage with lifecycle policies, a
4×T4-GPU batched inference pool, and a full DR plan with RPO/RTO targets. None
of this is implemented in code — the running system today is one Kafka broker,
one Redis node, CPU-only MobileNetV2, and no S3 wiring at all. That's
completely normal for a document titled around "Interview Talking Points," but
it's worth stating plainly so the design doc and the running pipeline are never
confused for the same thing.

---

## Known gaps and recommended improvements

Ordered roughly by how much they change what the project actually demonstrates.

#### 1. `perceptual_hash.py` is fully built and never used
`dhash`/`ahash`/`phash`/`hamming_distance`/`are_similar` exist, are individually
correct, and have zero call sites. Wiring `are_similar` into
`EmbeddingCache.lookup()` as a fallback (check a small window of recent hashes
within Hamming distance ≤ N before declaring a miss) would make the marquee
"perceptual hash caching" claim literally true instead of approximately true.
This is the highest-leverage single change available.

#### 2. `S3Handler` is fully built and never called
`upload_model_artifact`, `upload_processing_log`, and `upload_embeddings_batch`
are complete and untested-but-correct-looking. Nothing in `processor/flink_job.py`
calls any of them. Batch-uploading processed embeddings periodically from
`VoyagerStreamProcessor` would close the gap between the architecture diagram
and the code.

#### 3. The docker-compose Flink cluster is never actually used
`--mode flink` runs an embedded PyFlink MiniCluster inside the Python process
(hence the distinct `rest.port=8082`); it never submits a job to
`flink-jobmanager:8081`. Either add a real submission path (package the job,
`flink run` against the cluster) or remove the `flink-jobmanager`/`flink-taskmanager`
services and document `--mode flink` honestly as "local Flink runtime for
exercising the DataStream API," not a cluster deployment.

#### 4. ~~`multi_camera_producer.py` duplicates and contradicts `kafka_producer.py`~~ — docstring fixed
Its docstring said 4 cameras; its code hardcodes 6 (`virat_test_01–06.mp4`) at
a fixed 33ms interval, ignoring `config/settings.py` entirely. The docstring
now states this accurately. The duplication and the hardcoded paths are still
there — worth deleting this file in favor of the settings-driven
`run_multi_camera_producer()` if that's ever a priority.

#### 5. Missing test coverage
No tests for `cache_handler.py`, `model_handler.py`, `s3_handler.py`,
`metrics_exporter.py`, or `flink_job.py`. Restoring `test_cache.py` first would
cover the most-connected module in the codebase.

#### 6. ~~The only measured cache hit rate is 75%, not 40%~~ — README reworded
`test_cache_hit_simulation()`'s default parameters mathematically guarantee a
75% hit rate. The README's "40%+" has been reworded to state both numbers and
where each comes from. Running the load test against the real
`virat_test_*.mp4` footage in `data/sample_videos/` to get a realistic measured
number is still open.

#### 7. ~~Two Flink UIs, two ports, no explanation~~ — README clarified
README now states both ports and which Flink instance each belongs to.

#### 8. Dead configuration
`PROMETHEUS_PORT`, `GRAFANA_PORT`, and `METRICS_EXPORT_PORT` are declared in
`.env` but read by nothing — still open (`.env` isn't tracked by git, so
there's nothing to fix in the repo itself; worth pruning locally if you touch
`.env` next). `KAFKA_TOPIC_METRICS` has been removed from `KafkaSettings`.

#### 9. No Dockerfile for the application itself
`docker-compose.yml` containerizes only infrastructure; `producer`, `processor`,
and `inference` run on the host via a local venv. Fine for local dev, but worth
stating since the scaling doc's GPU pool and TaskManager fleet both assume
containerized app code that doesn't exist yet.

#### 10. ~~The throughput gauge recomputes a list on every frame~~ — fixed
`MetricsCollector.record_frame_processed` used to rebuild `self._throughput_window`
with a list comprehension on every call — O(n) per frame where n scales with
sustained FPS over the 5-second window. It's now a `collections.deque` with a
timestamp-based `popleft` loop, which is O(1) amortized per frame.

#### 11. ~~`ModelHandler` has no GPU path~~ — fixed
`_select_device()` now picks CUDA, then MPS, then CPU, and the model and
output tensor move accordingly. Verified on this machine: MPS is selected and
used automatically. Still just single-frame inference on one device, not the
scaling doc's batched multi-GPU pool.

#### 12. ~~Minor: unused `import hashlib` in `frame_processor.py`~~ — removed

---

## File map

| Path | Documented in |
|---|---|
| `config/settings.py` | [Configuration surface](#configuration-surface) |
| `producer/kafka_producer.py` | [Producer layer](#producer-layer-producer) |
| `producer/multi_camera_producer.py` | [Producer layer](#producer-layer-producer), [Gap #4](#4-multi_camera_producerpy-duplicates-and-contradicts-kafka_producerpy) |
| `producer/sample_video_generator.py` | [Producer layer](#producer-layer-producer) |
| `processor/frame_processor.py` | [Frame decoding & metadata](#frame-decoding--metadata-processorframe_processorpy) |
| `processor/cache_handler.py` | [The cache layer](#the-cache-layer--what-perceptual-hash-caching-actually-means-here) |
| `processor/flink_job.py` | [The frame lifecycle](#the-frame-lifecycle-precisely), [Orchestration modes](#orchestration-modes-processorflink_jobpy) |
| `inference/model_handler.py` | [AI inference](#ai-inference-inferencemodel_handlerpy) |
| `inference/perceptual_hash.py` | [The cache layer](#the-cache-layer--what-perceptual-hash-caching-actually-means-here), [Gap #1](#1-perceptual_hashpy-is-fully-built-and-never-used) |
| `monitoring/metrics_exporter.py` | [Metrics](#metrics-monitoringmetrics_exporterpy) |
| `storage/s3_handler.py` | [Storage](#storage-storages3_handlerpy), [Gap #2](#2-s3handler-is-fully-built-and-never-called) |
| `docker-compose.yml` | [Local infrastructure topology](#local-infrastructure-topology) |
| `loadtest/scale_design.md` | [The 10TB/day scaling design](#the-10tbday-scaling-design--whats-real-vs-whats-a-plan) |
| `loadtest/stress_test.py` | [Testing & verification](#testing--verification) |
