import torch
import numpy as np
import librosa
from collections import Counter
from instrument_classifier.model import CNNAudioClassifier
from loguru import logger


def process_audio_segment(audio_data: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
    """Process a single 10-second audio segment into a spectrogram."""
    logger.debug("Converting audio segment to mel spectrogram")
    # Generate mel spectrogram using the same parameters as in data.py
    S = librosa.feature.melspectrogram(y=audio_data.astype(float), sr=sample_rate, n_mels=128, fmax=8000)
    S_DB = librosa.power_to_db(S, ref=np.max)
    # Normalize from [-80, 0] to [-1, 1] range
    S_DB = (S_DB + 40) / 40
    logger.debug("Audio segment processed successfully")
    return S_DB


def predict_single_segment(model: CNNAudioClassifier, spectrogram: np.ndarray) -> int:
    """Make a prediction for a single spectrogram."""
    logger.debug("Making prediction for single spectrogram segment")
    model.eval()
    with torch.no_grad():
        # Add batch and channel dimensions
        spectrogram_tensor = torch.tensor(spectrogram).unsqueeze(0).unsqueeze(0).float()
        output = model(spectrogram_tensor)
        predicted_class = output.argmax(dim=1).item()
    logger.debug(f"Segment prediction: class {predicted_class}")
    return predicted_class


def process_audio_file(audio_path: str, model: CNNAudioClassifier) -> int:
    """Process an audio file and return the predicted class.

    If the clip is shorter than 10s, pad with zeros.
    If longer than 10s, split into 10s segments and use majority voting.
    """
    logger.info(f"Processing audio file: {audio_path}")
    try:
        model.eval()
    except Exception as e:
        logger.error(f"Error setting model to evaluation mode: {str(e)}")
        raise

    # Constants
    TARGET_DURATION = 10
    TARGET_SR = 44100
    TARGET_SAMPLES = TARGET_DURATION * TARGET_SR

    # Load audio with original sample rate
    logger.debug("Loading audio file")
    audio_data, original_sr = librosa.load(audio_path, sr=None)
    logger.debug(f"Audio loaded - Original sample rate: {original_sr}, Length: {len(audio_data)} samples")

    # Resample if necessary
    if original_sr != TARGET_SR:
        logger.debug(f"Resampling audio from {original_sr}Hz to {TARGET_SR}Hz")
        audio_data = librosa.resample(y=audio_data, orig_sr=original_sr, target_sr=TARGET_SR)

    # Handle different audio lengths
    if len(audio_data) < TARGET_SAMPLES:
        logger.debug(f"Audio shorter than {TARGET_DURATION}s, padding with zeros")
        # Pad with zeros if too short
        padding = TARGET_SAMPLES - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), mode="constant")
        spectrogram = process_audio_segment(audio_data, TARGET_SR)
        prediction = predict_single_segment(model, spectrogram)
        logger.info(f"Final prediction for short audio: class {prediction}")
        return prediction

    else:
        logger.debug(f"Audio longer than {TARGET_DURATION}s, using segment-based prediction")
        # Split into 10-second segments if longer
        predictions = []
        for start_idx in range(0, len(audio_data), TARGET_SAMPLES):
            segment = audio_data[start_idx : start_idx + TARGET_SAMPLES]
            if len(segment) == TARGET_SAMPLES:  # Only process complete segments
                logger.debug(f"Processing segment from {start_idx} to {start_idx + TARGET_SAMPLES} samples")
                spectrogram = process_audio_segment(segment, TARGET_SR)
                pred = predict_single_segment(model, spectrogram)
                predictions.append(pred)

        # Return most common prediction (majority voting)
        if predictions:
            return Counter(predictions).most_common(1)[0][0]
        else:
            raise ValueError("No valid predictions could be made from the audio file")


def load_model(model_path: str = "models/best_cnn_audio_classifier.pt") -> CNNAudioClassifier:
    """Load the trained model."""
    for i in [1, 3, 5, 7]:
        try:
            model = CNNAudioClassifier(num_classes=4, input_channels=1, num_layers=i)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    raise
