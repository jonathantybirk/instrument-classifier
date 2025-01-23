import torch
import pytest
from src.instrument_classifier.model import CNNAudioClassifier


def test_model_initialization():
    """Test model initialization with different parameters."""
    # Test default initialization
    model = CNNAudioClassifier()
    assert isinstance(model, CNNAudioClassifier)

    # Test with custom parameters
    model = CNNAudioClassifier(num_classes=10, input_channels=2)
    assert isinstance(model, CNNAudioClassifier)


def test_forward_pass():
    """Test forward pass with different input sizes."""
    model = CNNAudioClassifier(num_classes=4, input_channels=1)

    # Test with small input
    batch_size, n_mels, seq_len = 2, 64, 32
    x = torch.randn(batch_size, 1, n_mels, seq_len)
    output = model(x)
    assert output.shape == (batch_size, 4)

    # Test with larger input
    batch_size, n_mels, seq_len = 16, 128, 64
    x = torch.randn(batch_size, 1, n_mels, seq_len)
    output = model(x)
    assert output.shape == (batch_size, 4)


def test_output_properties():
    """Test properties of model outputs."""
    model = CNNAudioClassifier(num_classes=3)
    x = torch.randn(4, 1, 64, 48)

    # Test output type and shape
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32
    assert output.shape == (4, 3)

    # Test that output contains valid logits (no NaN or inf)
    assert torch.isfinite(output).all()


def test_model_parameters():
    """Test model parameter properties."""
    model = CNNAudioClassifier()

    # Verify model has learnable parameters
    assert sum(p.numel() for p in model.parameters()) > 0

    # Test that gradients can flow
    x = torch.randn(2, 1, 64, 48)
    output = model(x)
    loss = output.sum()
    loss.backward()

    # Verify gradients are computed
    for param in model.parameters():
        assert param.grad is not None


def test_invalid_inputs():
    """Test model behavior with invalid inputs."""
    model = CNNAudioClassifier()

    # Test with wrong number of dimensions
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 64, 48))  # Missing channel dimension

    # Test with wrong number of channels
    with pytest.raises(RuntimeError):
        model(torch.randn(2, 2, 64, 48))  # 2 channels instead of 1
