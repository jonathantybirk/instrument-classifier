"""Module for training the instrument classifier model.

This module handles the training process of the CNN-based audio classifier,
including data loading, model initialization, and the training loop.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from torch.profiler import profile

from data import InstrumentDataset
from model import CNNAudioClassifier


def train_model(num_epochs: int = 5, profiler: Optional[profile] = None) -> None:
    """Train the CNN audio classifier model.

    Args:
        num_epochs: Number of training epochs. Defaults to 5.
        profiler: Optional profiler instance for performance analysis.

    The function:
    1. Initializes the dataset and dataloader
    2. Sets up the model, loss function, and optimizer
    3. Trains the model for the specified number of epochs
    4. Saves the trained model weights
    """
    logging.info("Initializing training process")

    # Example dataset and DataLoader
    dataset = InstrumentDataset(
        data_path=Path("data/processed/train"),  # Adjust as needed
        metadata_path=Path("data/raw/metadata_train.csv"),  # Adjust as needed
    )
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Instantiate the model
    model = CNNAudioClassifier(num_classes=4, input_channels=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data = data.clone().detach().unsqueeze(1).float()  # Example reshape
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Step the profiler if it's active
            if profiler is not None:
                profiler.step()

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), Path("models/cnn_audio_classifier.pt"))


if __name__ == "__main__":
    train_model()
