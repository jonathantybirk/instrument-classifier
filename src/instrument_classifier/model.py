import torch
from torch import nn
import logging
from hydra import initialize, compose
from omegaconf import DictConfig


class CNNAudioClassifier(nn.Module):
    """A CNN-based audio classifier that supports variable-length spectrogram inputs."""

    def __init__(
        self,
        num_layers: int = 3,
        num_classes: int = 4,
        input_channels: int = 1,
        kernel_size: int = 3,
        dropout_rate: float = 0.5,
    ) -> None:
        """
        Initialize the model.

        Args:
            num_classes (int): Number of target classes.
            input_channels (int): Number of input channels (e.g., 1 for single-channel spectrograms).
        """
        logging.info("Initializing CNNAudioClassifier")
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        layers = []
        if num_layers in [3, 5, 7]:
            layers.extend(
                [
                    nn.Conv2d(8, 16, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                    nn.Conv2d(16, 32, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                ]
            )

        if num_layers in [5, 7]:
            layers.extend(
                [
                    nn.Conv2d(32, 64, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                    nn.Conv2d(64, 64, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                ]
            )

        if num_layers == 7:
            layers.extend(
                [
                    nn.Conv2d(64, 64, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                    nn.Conv2d(64, 64, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                ]
            )

        self.conv_layers = nn.Sequential(self.conv_layers, *layers)

        self.fc = nn.Sequential(nn.Flatten(), nn.LazyLinear(100), nn.ReLU(), nn.Linear(100, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, n_mels, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        x = self.conv_layers(x)
        x = self.dropout(x)
        return self.fc(x)


if __name__ == "__main__":
    # Initialize Hydra and load the configuration
    with initialize(config_path="../../conf", version_base=None):
        cfg: DictConfig = compose(config_name="config")

    # Set logging level from config
    logging.basicConfig(level=getattr(logging, cfg.logging.level))
    logging.info("Loaded configuration")

    # Model initialization
    model = CNNAudioClassifier(
        num_classes=cfg.model.num_classes,
        input_channels=cfg.model.input_channels,
    )

    # Test forward pass with dummy input
    dummy_input = torch.randn(
        cfg.dummy_input.batch_size,
        cfg.model.input_channels,
        cfg.dummy_input.n_mels,
        cfg.dummy_input.seq_len,
    )
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Print model architecture
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
