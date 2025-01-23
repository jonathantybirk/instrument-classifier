"""Module for training the instrument classifier model.

This module handles the training process of the CNN-based audio classifier,
including data loading, model initialization, and the training loop.
"""

from pathlib import Path
from typing import Optional
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.profiler import profile
from instrument_classifier.data import InstrumentDataset
from instrument_classifier.model import CNNAudioClassifier
import random
import numpy as np
import wandb
from loguru import logger

# Configure loguru to write logs to a file and not to the console
logger.remove()  # Remove the default logger
logger.add("logging/training.log", rotation="100 MB")

logger.info("Loguru logger initialized")


def train_model(
    num_epochs: int = 50,  # Increased default epochs since we have early stopping
    patience: int = 5,  # Number of epochs to wait before early stopping
    val_split: float = 0.2,  # Validation set size as fraction of total data
    profiler: Optional[profile] = None,
) -> None:
    # Set all random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)  # For CUDA GPU
    torch.cuda.manual_seed_all(42)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create a generator for reproducible data splits
    generator = torch.Generator().manual_seed(42)
    
    wandb.init(
        project="instrument_classifier",
        config={
            "num_epochs": num_epochs,
            "patience": patience,
            "val_split": val_split,
            "seed": 42
        },
    )

    """Train the CNN audio classifier model.

    Args:
        num_epochs: Number of training epochs. Defaults to 50.
        patience: Number of epochs to wait before early stopping. Defaults to 5.
        val_split: Fraction of the dataset to use as validation. Defaults to 0.2.
        profiler: Optional profiler instance for performance analysis.

    The function:
    1. Initializes the dataset and dataloaders
    2. Sets up the model, loss function, and optimizer
    3. Trains the model for the specified number of epochs
    4. Saves the trained model weights
    5. Creates and saves a loss plot
    """
    wandb.log({"message": "Initializing training process"})

    # Load the full dataset
    dataset = InstrumentDataset(
        data_path=Path("data/processed/train"),
        metadata_path=Path("data/processed/metadata_train.csv"),
    )

    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Use the same generator for DataLoader to ensure reproducible shuffling
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator, worker_init_fn=lambda x: torch.manual_seed(42))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CNNAudioClassifier(num_classes=4, input_channels=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize tracking variables
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_samples = 0
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data = data.clone().detach().unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate weighted loss
            batch_size = data.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

            if profiler is not None:
                profiler.step()

            # Log batch loss to the designated file for specific batches
            if batch_idx + 1 in [1, 10, 20, 30, 40, 50, 60]:
                logger.info(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / total_train_samples
        train_losses.append(avg_train_loss)
        wandb.log({"train_loss": avg_train_loss})
        logger.warning(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.clone().detach().unsqueeze(1).float()
                outputs = model(data)
                loss = criterion(outputs, labels)

                # Accumulate weighted loss
                batch_size = data.size(0)
                total_val_loss += loss.item() * batch_size
                total_val_samples += batch_size

        avg_val_loss = total_val_loss / total_val_samples
        val_losses.append(avg_val_loss)

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        print(f"Epoch [{epoch+1}/{num_epochs}], " f"Train Loss: {avg_train_loss:.4f}, " f"Val Loss: {avg_val_loss:.4f}")

        # Check if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping check
        if patience - patience_counter <= 3:
            logger.warning(
                f"Warning: Early stopping will be triggered in {patience - patience_counter} epochs if no improvement in validation loss T-T"
            )
            break

    # Save the best model
    torch.save(best_model_state, Path("models/best_cnn_audio_classifier.pt"))
    logger.warning("Training has ended and the model has been saved in models/best_cnn_audio_classifier.pt")

    # Create and save the loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True)

    # Create reports/figures directory if it doesn't exist
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.savefig(figures_dir / "training_validation_loss.png")
    plt.close()

    # Save the loss data
    loss_data = {"train_losses": train_losses, "val_losses": val_losses}
    with open(figures_dir / "loss_data.json", "w") as f:
        json.dump(loss_data, f)


if __name__ == "__main__":
    train_model()
