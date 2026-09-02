"""Tests for the AI model handler."""
import numpy as np
import pytest

from inference.model_handler import ModelHandler


class TestSelectDevice:
    def test_prefers_cuda(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
        assert ModelHandler._select_device() == "cuda"

    def test_falls_back_to_mps_without_cuda(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
        assert ModelHandler._select_device() == "mps"

    def test_falls_back_to_cpu_without_any_accelerator(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
        assert ModelHandler._select_device() == "cpu"


class TestLazyLoading:
    def test_model_not_loaded_until_first_use(self):
        handler = ModelHandler()
        assert handler._model is None

    def test_ensure_loaded_only_loads_once(self, monkeypatch):
        handler = ModelHandler()
        calls = []
        monkeypatch.setattr(handler, "_load_mobilenet", lambda: calls.append(1) or setattr(handler, "_model", object()))

        handler._ensure_loaded()
        handler._ensure_loaded()

        assert len(calls) == 1

    def test_unknown_model_name_raises(self):
        handler = ModelHandler(model_name="not-a-real-model")
        with pytest.raises(ValueError, match="Unknown model"):
            handler._ensure_loaded()


class TestAvgInferenceMs:
    def test_zero_before_any_inference(self):
        handler = ModelHandler()
        assert handler.avg_inference_ms == 0.0


class TestGenerateEmbedding:
    def test_real_mobilenet_inference_produces_1280_dim_embedding(self):
        handler = ModelHandler()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        embedding = handler.generate_embedding(frame)

        assert embedding.shape == (1280,)
        assert embedding.dtype == np.float32
        assert handler.avg_inference_ms > 0.0
