import pytest
import torch
from pathlib import Path
import tempfile
from omegaconf import OmegaConf
from instrument_classifier.train import train_model
from instrument_classifier.data import InstrumentDataset
import wandb


class MockWandb:
    """Mock class for wandb with all required functionality"""

    def __init__(self):
        self.config = {}
        self.summary = {}
        self.logged_data = []

    def init(self, **kwargs):
        self.config.update(kwargs)
        return self

    def log(self, data):
        self.logged_data.append(data)


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock wandb to avoid actual cloud logging during tests"""
    mock = MockWandb()

    # Mock all required wandb functionality
    monkeypatch.setattr(wandb, "init", mock.init)
    monkeypatch.setattr(wandb, "log", mock.log)
    monkeypatch.setattr(wandb, "summary", mock.summary)

    return mock


@pytest.fixture
def test_config():
    """Create a minimal test configuration"""
    config = {
        "model": {
            "num_classes": 4,  # Keep default number of classes
            "input_channels": 1,
            "num_layers": 2,
            "kernel_size": 3,
            "dropout_rate": 0.1,
        },
        "training": {
            "batch_size": 2,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "val_split": 0.25,  # Changed to ensure at least 2 validation samples
            "patience": 3,
            "seed": 42,
        },
        "data": {"train_data_path": "data/processed/train", "train_metadata_path": "data/processed/metadata_train.csv"},
        "paths": {"model_save": "models/test_model.pth", "figures": "reports/figures"},
        "wandb": {"project": "test-project", "entity": "test-entity"},
    }
    return OmegaConf.create(config)


def test_minimal_train(mock_wandb, test_config, monkeypatch):
    """Test training with minimal data for just a few steps"""

    # Create a minimal dummy dataset that properly implements InstrumentDataset interface
    class MinimalDataset(InstrumentDataset):
        def __init__(self, data_path, metadata_path):
            # Mock the metadata and class mapping
            self.classes = ["violin", "piano", "guitar", "drums"]
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            # Create dummy data - 8 samples, 2 for each class
            self.data = [(torch.randn(128, 862), idx % len(self.classes)) for idx in range(8)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Mock the dataset loading
    monkeypatch.setattr("instrument_classifier.train.InstrumentDataset", MinimalDataset)

    # Create temporary directories for model and figure saving
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_config.paths.model_save = str(temp_path / "test_model.pth")
        test_config.paths.figures = str(temp_path / "figures")

        # Run training
        train_model(test_config)

        # Verify outputs
        assert Path(test_config.paths.model_save).exists(), "Model file was not saved"
        assert Path(test_config.paths.figures).exists(), "Figures directory was not created"
        assert (Path(test_config.paths.figures) / "training_validation_loss.png").exists(), "Loss plot was not saved"
        assert (Path(test_config.paths.figures) / "loss_data.json").exists(), "Loss data was not saved"

        # Verify wandb logging occurred
        assert len(mock_wandb.logged_data) > 0, "No data was logged to wandb"
        assert "best_validation_loss" in mock_wandb.summary, "Best validation loss not logged to wandb summary"
