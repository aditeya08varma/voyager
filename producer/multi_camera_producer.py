"""
Multi-Camera Kafka Producer for Voyager

Spawns 6 concurrent camera simulators, each reading one of the hardcoded
video_sources below at a fixed 33ms interval (~30 FPS). Unlike
kafka_producer.run_multi_camera_producer(), this entry point ignores
config/settings.py's NUM_SIMULATED_CAMERAS, VIDEO_SOURCE, and
FRAME_INTERVAL_MS entirely.
"""
from __future__ import annotations

import os
import sys
import time
import structlog

# Ensure we can import from the root Voyager directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import settings
from producer.kafka_producer import create_producer, CameraSimulator

log = structlog.get_logger(__name__)

def main():
    log.info("Starting Multi-Camera Producer Setup...")
    
    # 1. Initialize the single Kafka connection
    producer = create_producer()
    
    # 2. Define your 4 video sources (Change these if you downloaded 4 distinct files)
    video_sources = [
        "./data/sample_videos/virat_test_01.mp4",
        "./data/sample_videos/virat_test_02.mp4",
        "./data/sample_videos/virat_test_03.mp4",
        "./data/sample_videos/virat_test_04.mp4",
        "./data/sample_videos/virat_test_05.mp4",
        "./data/sample_videos/virat_test_06.mp4"
    ]
    
    cameras: list[CameraSimulator] = []

    # 3. Spin up a separate thread for each camera
    for i, video_path in enumerate(video_sources):
        camera_id = f"CAM-00{i+1}"
        
        if not os.path.exists(video_path):
            log.error(f"Video file not found: {video_path}")
            continue

        cam = CameraSimulator(
            camera_id=camera_id,
            video_path=video_path,
            producer=producer,
            topic=settings.kafka.topic_frames,
            frame_interval_ms=33, # roughly 30 FPS
        )
        cameras.append(cam)
        cam.start()
        log.info(f"Started {camera_id} reading from {video_path}")

    # 4. Keep the main script alive while the background threads run
    try:
        log.info("All cameras streaming. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        log.info("Shutting down cameras gracefully...")
        for cam in cameras:
            cam.stop()
        for cam in cameras:
            cam.join()
        log.info("Shutdown complete.")

if __name__ == "__main__":
    main()