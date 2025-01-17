import torch
import logging
from torch.utils.data import DataLoader
from model import CNNAudioClassifier
from data import InstrumentDataset


def evaluate_model():
    logging.info("Loading dataset for evaluation")
    dataset = InstrumentDataset(
        data_path="data/processed/test",  # Adjust as needed
        metadata_path="data/raw/metadata_test.csv",  # Adjust as needed
    )
    eval_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    logging.info("Loading model weights")
    model = CNNAudioClassifier(num_classes=4, input_channels=1)
    model.load_state_dict(torch.load("models/cnn_audio_classifier.pt"))
    model.eval()

    # Placeholder loop for evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in eval_loader:
            data = torch.tensor(data).unsqueeze(1).float()  # Example reshape
            outputs = model(data)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    accuracy = correct / total if total else 0
    logging.info(f"Evaluation accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    evaluate_model()
