#!/usr/bin/env bash
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║   Voyager Pipeline — Setup Script        ║"
echo "╚══════════════════════════════════════════╝"

# Check prerequisites
for cmd in docker docker-compose python3; do
    if ! command -v $cmd &>/dev/null; then
        echo "ERROR: $cmd not found. Please install it first."
        exit 1
    fi
done

echo "[1/5] Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet -r requirements.txt

echo "[2/5] Generating sample video..."
python -c "from producer.sample_video_generator import generate_sample_video; generate_sample_video()"

echo "[3/5] Starting Docker services..."
docker-compose up -d

echo "[4/5] Waiting for services to be ready..."
sleep 15

echo "[5/5] Verifying services..."
echo -n "  Kafka:      " && (docker exec voyager-kafka kafka-topics --bootstrap-server localhost:29092 --list &>/dev/null && echo "OK" || echo "WAITING")
echo -n "  Redis:      " && (docker exec voyager-redis redis-cli ping 2>/dev/null || echo "WAITING")
echo -n "  Prometheus: " && (curl -sf http://localhost:9090/-/ready &>/dev/null && echo "OK" || echo "WAITING")
echo -n "  Grafana:    " && (curl -sf http://localhost:3000/api/health &>/dev/null && echo "OK" || echo "WAITING")

echo ""
echo "Setup complete! Quick start:"
echo "  Terminal 1:  python -m processor.flink_job --mode flink"
echo "  Terminal 2:  python -m producer.kafka_producer"
echo ""
echo "Dashboards:"
echo "  Flink:      http://localhost:8082"
echo "  Grafana:    http://localhost:3000  (admin/voyager)"
echo "  Prometheus: http://localhost:9090"
