# Voyager — 10TB/Day Scaling Architecture

## Executive Summary

This document outlines the technical architecture for scaling the Voyager
video processing pipeline from a local development setup to handling
**10TB of video data per day** across hundreds of camera feeds.

---

## 1. Kafka Scaling Strategy

### Current State
- Single broker, 6 partitions, ~10 FPS per camera

### Target: 10TB/day Ingestion
- **Broker topology**: 5-node cluster across 3 AZs
- **Partitioning**: 120 partitions (one per camera group)
  - Key: `camera_id` → ensures per-camera ordering
  - Partition assignment: consistent hashing
- **Replication factor**: 3 (min.insync.replicas=2)
- **Message sizing**: ~50KB/frame × 30fps × 100 cameras ≈ 150MB/s sustained
- **Retention**: 4 hours hot (SSD), 7 days warm via tiered storage to S3
- **Compression**: LZ4 (best throughput/ratio for JPEG payloads)

### Capacity Math
```
10TB/day = 115MB/s sustained
Per broker: ~25MB/s write + 50MB/s read (replication + consumers)
5 brokers × 200MB/s network = 1GB/s capacity (4x headroom)
```

---

## 2. Flink Cluster Sizing

### Current State
- Local mini-cluster, parallelism=2

### Target: Sub-50ms at Scale
- **JobManager**: 2 HA instances (ZooKeeper-based leader election)
- **TaskManagers**: 8 nodes, 4 slots each = 32 parallel operators
- **Checkpointing**: Incremental, RocksDB state backend, S3 checkpoint storage
- **Tuning**:
  - `buffer-timeout: 1ms` (latency)
  - `network.memory.fraction: 0.2`
  - `taskmanager.memory.process.size: 8GB`
  - `execution.checkpointing.interval: 10000`
  - `execution.checkpointing.min-pause: 5000`

### Parallelism Design
```
120 Kafka partitions → 32 Flink slots
Each slot handles ~4 partitions
Per-slot throughput: ~15MB/s decode + hash + cache check
```

---

## 3. Redis Cluster Architecture

### Current State
- Single node, 512MB, LRU eviction

### Target: 40%+ Cache Hit Rate at Scale
- **Topology**: Redis Cluster, 6 masters + 6 replicas across 3 AZs
- **Memory**: 32GB per node = 192GB total cache capacity
- **Key design**: `voyager:emb:{content_hash}` (16-char hex = 64-bit dHash)
- **TTL strategy**:
  - Background frames (high hit rate): 4 hours
  - Dynamic frames (low reuse): 30 minutes
  - Adaptive TTL based on hit frequency
- **Eviction**: `allkeys-lfu` (frequency-based, better than LRU for video)

### Cache Warming
- Pre-populate cache with known static backgrounds per camera on deployment
- Estimated warm cache hit rate: 55-65% (vs 40% cold start)

### Capacity Math
```
Embedding size: 1280 × 4 bytes = 5.12KB
Unique hashes per hour: ~50,000 (across 100 cameras)
Storage per hour: 50K × 5.12KB = 256MB
192GB capacity = ~750 hours of unique embeddings
```

---

## 4. S3 Storage Architecture

### Bucket Layout
```
voyager-frames/
  ├── raw/{YYYY}/{MM}/{DD}/{camera_id}/{HH}/
  ├── processed/{YYYY}/{MM}/{DD}/{camera_id}/{HH}/
  └── embeddings/{YYYY}/{MM}/{DD}/{camera_id}/batch_{ts}.jsonl

voyager-artifacts/
  ├── models/{model_name}/{version}/
  └── configs/{deployment_id}/

voyager-logs/
  └── pipeline/{YYYY}/{MM}/{DD}/{HH}/
```

### Sharding Strategy
- **Time-based partitioning**: hourly prefixes prevent hot-partition issues
- **Camera-based sub-sharding**: parallel writes per camera
- **Lifecycle policies**:
  - Raw frames: 30 days → Glacier → delete at 1 year
  - Processed metadata: 90 days → IA → Glacier at 1 year
  - Embeddings: 180 days → IA
  - Model artifacts: indefinite

### Cost Estimate (10TB/day)
```
Storage (S3 Standard):  300TB/month × $0.023/GB = $6,900/month
PUT requests:           ~86M/day × $0.005/1K = $430/month
GET requests:           ~200M/day × $0.0004/1K = $80/month
Transfer:               internal (free with VPC endpoint)
Total S3:               ~$7,400/month
```

---

## 5. AI Inference Scaling

### Current State
- MobileNetV2 on CPU, ~20ms/frame

### Target: Batched GPU Inference
- **GPU pool**: 4× NVIDIA T4 instances (p3.2xlarge or g4dn.xlarge)
- **Batched inference**: Collect 32 frames → single GPU forward pass
  - Latency: ~5ms per batch = 0.15ms per frame amortized
  - Throughput: ~6,400 inferences/second per GPU
- **Model versioning**: S3-backed model registry with A/B deployment
- **Fallback**: CPU inference pool for overflow (auto-scaling group)

### With Cache (40% hit rate)
```
Total frames/day: 10TB ÷ 50KB/frame ≈ 200M frames
Cache hits (40%): 80M frames → skip inference entirely
Inference needed: 120M frames ÷ 86400s = 1,389 frames/s
4 GPUs × 6,400 fps = 25,600 fps capacity (18x headroom)
```

---

## 6. Monitoring at Scale

### Metrics Cardinality
- Labels: `camera_id` (100+), `region`, `model_version`
- Estimated active series: ~5,000
- Use Thanos or Cortex for long-term Prometheus storage

### Alerting Rules
- P99 latency > 50ms for 5 minutes → PagerDuty
- Cache hit rate < 30% for 10 minutes → Slack
- Kafka consumer lag > 10,000 → PagerDuty
- GPU utilization > 80% sustained → auto-scale trigger

---

## 7. Disaster Recovery

### RPO/RTO Targets
- **RPO**: 5 minutes (Kafka replication + S3 cross-region)
- **RTO**: 15 minutes (Flink savepoint restore + Kafka offset reset)

### Multi-Region Strategy
- Active-passive with automated failover
- Kafka MirrorMaker 2.0 for cross-region replication
- S3 Cross-Region Replication for model artifacts
- Redis Global Datastore for cache synchronization

---

## 8. Interview Talking Points

1. **"How do you handle 10TB/day?"**
   → Time-partitioned S3 sharding + Kafka tiered storage eliminates hot spots

2. **"How do you keep latency under 50ms?"**
   → Redis cache (40%+ hit rate) + batched GPU inference + Flink buffer timeout tuning

3. **"What happens when a node fails?"**
   → Kafka replication factor 3 + Flink exactly-once checkpointing + Redis Cluster auto-failover

4. **"How do you optimize inference costs?"**
   → Perceptual hashing for near-duplicate detection eliminates 40-60% of inference calls

5. **"How do you monitor at scale?"**
   → Custom Prometheus metrics with Thanos for long-term storage, Grafana dashboards with SLO-based alerting
