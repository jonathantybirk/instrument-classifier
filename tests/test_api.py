import pytest
from fastapi.testclient import TestClient
import os
from pathlib import Path
import tempfile
import numpy as np
from instrument_classifier.api import app, dataset


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance with proper lifespan handling"""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200, "Health check status code is not 200"
    assert response.json()["status"] == "healthy", "Status is not healthy"
    assert "model_loaded" in response.json(), "Incorrect response healthcheck format"
    assert response.json()["model_loaded"] is True, "Model failed to load during test client initialization"


def test_predict_with_real_audio_file(client):
    """Test prediction with audio files from the train dataset"""
    audio_path1 = "data/raw/train_submission/AR_Lick5_MBVDN.wav"
    audio_path2 = "data/raw/train_submission/0_oliver-colbentson_bwv1006_mov3.wav"

    for audio_path in [audio_path1, audio_path2]:
        with open(audio_path, "rb") as f:
            files = {"file": ("test.wav", f, "audio/wav")}
            response = client.post("/predict", files=files)
            assert response.status_code == 200
            prediction = response.json()
            assert "predicted_instrument" in prediction
            assert isinstance(prediction["predicted_instrument"], str)
            print(f"Predicted instrument for {audio_path}: {prediction['predicted_instrument']}")
