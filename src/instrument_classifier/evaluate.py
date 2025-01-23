import torch
from loguru import logger
from torch.utils.data import DataLoader
from model import CNNAudioClassifier
from data import InstrumentDataset


logger.remove()  # Remove the default logger
logger.add("output/logging/evaluation.log", rotation="100 MB")
logger.info("Evaluation logger initialized")
print("Evaluating model... check /logging/evaluation.log for progress and errors")


def evaluate_model():
    logger.info("Loading dataset for evaluation")
    dataset = InstrumentDataset(
        data_path="data/processed/test",  # Adjust as needed
        metadata_path="data/raw/metadata_test.csv",  # Adjust as needed
    )
    eval_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    logger.info("Loading model weights")
    model = CNNAudioClassifier(num_classes=4, input_channels=1)
    model.load_state_dict(torch.load("models/best_cnn_audio_classifier.pt"))
    model.eval()

    # Placeholder loop for evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in eval_loader:
            data = data.clone().detach().unsqueeze(1).float()  # More efficient tensor construction
            outputs = model(data)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    accuracy = correct / total if total else 0
    logger.warning(f"Evaluation accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    evaluate_model()
