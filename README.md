# Voyager — High-Throughput Video Processing Pipeline

Real-time video frame processing pipeline with AI inference caching,
built on Kafka, Apache Flink (PyFlink), Redis, and MobileNetV2.

For a full, source-verified technical deep dive — architecture diagrams, the
exact caching mechanism, and a list of known gaps between this README and the
code — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Architecture

```
Camera Feeds → Kafka → PyFlink → Redis Cache → AI Model → S3
                                    ↕
                              Prometheus → Grafana
```

## Key Metrics

- **Sub-50ms** per-frame processing latency (target; run `python -m loadtest.stress_test --test latency` for a measured number on your machine)
- **Cache-driven inference savings** — the cache does exact content-hash matching today, not fuzzy perceptual matching (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)); the bundled synthetic test (`--test cache`) measures ~75% hit rate at its default parameters, 40%+ is the production-scale target from `loadtest/scale_design.md`
- **1280-dim** MobileNetV2 embeddings
- **6-partition** Kafka topic with LZ4 compression

## Project Structure

```
Voyager/
├── config/          # Pydantic settings from .env
├── producer/        # Kafka producer + video simulator
├── processor/       # Flink job, frame decoder, Redis cache
├── inference/       # MobileNetV2 handler + perceptual hashing
├── monitoring/      # Prometheus metrics + Grafana dashboards
├── storage/         # S3 handler (LocalStack for dev)
├── loadtest/        # Stress tests + 10TB scaling design
├── tests/           # pytest suite
├── scripts/         # Setup automation
├── lib/             # Flink connector JARs
└── docker-compose.yml
```

## Quick Start

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker-compose up -d

# 2. Generate sample video
python -c "from producer.sample_video_generator import generate_sample_video; generate_sample_video()"

# 3. Run processor (Terminal 1)
python -m processor.flink_job --mode flink

# 4. Run producer (Terminal 2)
python -m producer.kafka_producer

# Or use standalone mode (no Flink JVM):
python -m processor.flink_job --mode standalone
```

## Dashboards

| Service    | URL                    | Credentials     |
|------------|------------------------|-----------------|
| Grafana    | http://localhost:3000  | admin / voyager |
| Flink UI (only while `--mode flink` is running) | http://localhost:8082 | — |
| Prometheus | http://localhost:9090  | —               |

> `--mode flink` runs its own embedded PyFlink MiniCluster with its UI on port
> 8082 — it does not submit to a separate Flink cluster; `docker-compose.yml`
> doesn't run one. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Cache Mechanism

```
Frame → dHash (8x8 DCT) → Redis Lookup
  ├── HIT  → Return cached 1280-dim embedding (< 1ms)
  └── MISS → MobileNetV2 inference → Store in Redis → Return
```

## Configuration

All settings via `.env` file. Key variables:

| Variable                 | Default                          |
|--------------------------|----------------------------------|
| KAFKA_BOOTSTRAP_SERVERS  | localhost:9092                    |
| REDIS_HOST               | localhost                         |
| FLINK_PARALLELISM        | 2                                 |
| MODEL_NAME               | mobilenetv2                       |
| NUM_SIMULATED_CAMERAS    | 4                                 |
| FRAME_INTERVAL_MS        | 100                               |

## Tests

```bash
pytest tests/ -v
python -m loadtest.stress_test --test all
```
