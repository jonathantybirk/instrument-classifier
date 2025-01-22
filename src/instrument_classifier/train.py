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

import wandb

def train_model(
    num_epochs: int = 50,  # Increased default epochs since we have early stopping
    patience: int = 5,  # Number of epochs to wait before early stopping
    val_split: float = 0.2,  # Validation set size as fraction of total data
    profiler: Optional[profile] = None,
) -> None:
    wandb.init(project="instrument_classifier", config={"num_epochs": num_epochs, "patience": patience, "val_split": val_split})

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
        metadata_path=Path("data/raw/metadata_train.csv"),
    )

    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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

        avg_train_loss = total_train_loss / total_train_samples
        train_losses.append(avg_train_loss)
        wandb.log({"train_loss": avg_train_loss})

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
        if patience_counter >= patience:
            wandb.log({"early_stopping_epoch": epoch + 1})
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save the best model
    torch.save(best_model_state, Path("models/best_cnn_audio_classifier.pt"))

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
