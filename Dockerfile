FROM python:3.12-slim

# libgl1/libglib2.0-0: needed by opencv-python-headless at import time.
# build-essential: apache-flink compiles a Cython extension
# (pyflink/fn_execution) at install time and needs gcc + libc headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# apache-flink needs this exact sequence on Python 3.12+ (see requirements.txt).
RUN pip install --no-cache-dir "setuptools<75" wheel && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY config/ config/
COPY producer/ producer/
COPY processor/ processor/
COPY inference/ inference/
COPY monitoring/ monitoring/
COPY storage/ storage/

# --mode flink additionally requires a JRE (Flink runs on the JVM) — not
# installed here. This image covers the producer and --mode standalone,
# which is everything runnable without the JVM.
CMD ["python", "-m", "processor.flink_job", "--mode", "standalone"]
