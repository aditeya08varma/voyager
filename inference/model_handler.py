"""
AI model handler for generating frame embeddings.

Supports MobileNetV2 (default, 1280-dim) with lazy loading.
The model is configured for feature extraction by replacing the
classifier head with an identity layer.
"""
from __future__ import annotations

import time

import numpy as np
import structlog
import torch
import torchvision.transforms as T
from PIL import Image

from config.settings import settings

log = structlog.get_logger(__name__)


class ModelHandler:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.model.name
        self._model = None
        self._transform = None
        self._device = "cpu"
        self._total_inferences = 0
        self._total_inference_ms = 0.0

    def _load_mobilenet(self):
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval()

        self._transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._model = model
        log.info("model_loaded", name="mobilenetv2", device=self._device, embedding_dim=1280)

    def _ensure_loaded(self):
        if self._model is None:
            if self._model_name == "mobilenetv2":
                self._load_mobilenet()
            else:
                raise ValueError(f"Unknown model: {self._model_name}")

    def generate_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Generate a feature embedding from a BGR numpy frame."""
        self._ensure_loaded()
        t0 = time.perf_counter()

        rgb = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        tensor = self._transform(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._model(tensor).squeeze(0).numpy()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_inferences += 1
        self._total_inference_ms += elapsed_ms

        if self._total_inferences % 50 == 0:
            avg = self._total_inference_ms / self._total_inferences
            log.info("inference_stats", total=self._total_inferences, avg_ms=round(avg, 2))

        return embedding

    @property
    def avg_inference_ms(self) -> float:
        if self._total_inferences == 0:
            return 0.0
        return self._total_inference_ms / self._total_inferences
