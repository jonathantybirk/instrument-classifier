from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from data import InstrumentDataset
from model import CNNAudioClassifier


def train_model():
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
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data = torch.tensor(data).unsqueeze(1).float()  # Example reshape
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), Path("models/cnn_audio_classifier.pt"))


if __name__ == "__main__":
    train_model()
