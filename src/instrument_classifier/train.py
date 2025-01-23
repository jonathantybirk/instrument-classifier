"""Module for training the instrument classifier model.

This module handles the training process of the CNN-based audio classifier,
including data loading, model initialization, and the training loop.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from instrument_classifier.data import InstrumentDataset
from instrument_classifier.model import CNNAudioClassifier
import random
import numpy as np
import wandb
from loguru import logger
import hydra
from omegaconf import DictConfig

# Configure loguru to write logs to a file and not to the console
logger.remove()  # Remove the default logger
logger.add("logging/training.log", rotation="100 MB")

logger.info("Loguru logger initialized")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train_model(cfg: DictConfig) -> None:
    """Train the CNN audio classifier model using Hydra configuration.

    Args:
        cfg: Hydra configuration object containing all parameters
    """
    # Set all random seeds for reproducibility
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)  # For CUDA GPU
    torch.cuda.manual_seed_all(cfg.training.seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create a generator for reproducible data splits
    generator = torch.Generator().manual_seed(cfg.training.seed)
    logger.warning(f"PyTorch generator seed set to {cfg.training.seed} for reproducible data splitting")

    # Initialize wandb with full configuration
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config={
            "model": dict(cfg.model),
            "training": dict(cfg.training),
            "data": dict(cfg.data),
            "paths": dict(cfg.paths),
        },
    )

    wandb.log({"message": "Initializing training process"})
    logger.info("Initializing training process")

    # Load the full dataset
    dataset = InstrumentDataset(
        data_path=Path(cfg.data.train_data_path),
        metadata_path=Path(cfg.data.train_metadata_path),
    )

    # Split into train and validation sets
    val_size = int(len(dataset) * cfg.training.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Use the same generator for DataLoader to ensure reproducible shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=lambda x: torch.manual_seed(cfg.training.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    model = CNNAudioClassifier(
        num_classes=cfg.model.num_classes,
        input_channels=cfg.model.input_channels,
        num_layers=cfg.model.num_layers,
        kernel_size=cfg.model.kernel_size,
        dropout_rate=cfg.model.dropout_rate,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Initialize tracking variables
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_samples = 0
        for batch_idx, (data, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        ):
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

            # Log batch loss to the designated file for specific batches
            if batch_idx % 10 == 1:
                logger.info(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / total_train_samples
        train_losses.append(avg_train_loss)
        wandb.log({"train_loss": avg_train_loss})
        logger.warning(f"Epoch [{epoch+1}/{cfg.training.num_epochs}], Train Loss: {avg_train_loss:.4f}")

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

        print(
            f"Epoch [{epoch+1}/{cfg.training.num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Check if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= cfg.training.patience:
            logger.warning(f"Early stopping triggered after {epoch+1} epochs")
            break
        elif cfg.training.patience - patience_counter <= 3:
            logger.warning(
                f"Warning: Early stopping will be triggered in {cfg.training.patience - patience_counter} epochs if no improvement in validation loss"
            )

    # Save the best model
    torch.save(best_model_state, Path(cfg.paths.model_save))
    logger.warning(f"Training has ended and the model has been saved in {cfg.paths.model_save}")

    # Create and save the loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True)

    # Create figures directory if it doesn't exist
    figures_dir = Path(cfg.paths.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.savefig(figures_dir / "training_validation_loss.png")
    plt.close()

    # Save the loss data
    loss_data = {"train_losses": train_losses, "val_losses": val_losses, "best_validation_loss": best_val_loss}
    with open(figures_dir / "loss_data.json", "w") as f:
        json.dump(loss_data, f)

    # Log final metrics to wandb summary
    wandb.summary["best_validation_loss"] = best_val_loss
    wandb.summary["final_train_loss"] = train_losses[-1]
    wandb.summary["final_val_loss"] = val_losses[-1]
    wandb.summary["total_epochs"] = len(train_losses)

    # Log final best validation loss
    logger.warning(f"Best validation loss achieved: {best_val_loss:.4f}")
    wandb.log({"best_validation_loss": best_val_loss})


if __name__ == "__main__":
    train_model()
