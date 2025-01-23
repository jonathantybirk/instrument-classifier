from pathlib import Path
import typer
from torch.utils.data import Dataset
import numpy as np
from scipy.io import wavfile
import librosa
import pandas as pd
import random
from tqdm import tqdm
from loguru import logger

logger.remove()  # Remove the default logger
logger.add("logging/preprocessing.log", rotation="100 MB")
logger.info("Loguru logger initialized")

class InstrumentDataset(Dataset):
    """Dataset class for audio classification."""

    def __init__(self, data_path: Path, metadata_path: Path) -> None:
        self.data_path = data_path
        self.metadata = pd.read_csv(metadata_path)
        self.classes = self.metadata["Class"].unique()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.metadata)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        row = self.metadata.iloc[index]
        spectrogram_path = self.data_path / f"{Path(row['FileName']).stem}.npy"
        label = self.class_to_idx[row["Class"]]

        # Load spectrogram
        spectrogram = np.load(spectrogram_path)
        return spectrogram, label

def preprocess(raw_data_path: Path, output_folder: Path, random_seed: int = 42) -> None:
    """Preprocess the raw audio files and save spectrograms."""
    print("Preprocessing data... check /logging/preprocessing.log for progress and errors")
    logger.warning("Preprocessing started")
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Define target duration in seconds and sample rate
    TARGET_DURATION = 10
    SAMPLE_RATE = 44100  # Standard audio sample rate
    TARGET_SAMPLES = TARGET_DURATION * SAMPLE_RATE
    
    for type in ["train", "test"]:
        # Load metadata
        metadata_path = raw_data_path / f"metadata_{type}.csv"
        metadata = pd.read_csv(metadata_path)

        # Log when starting to process train and test data
        logger.warning(f"Preprocessing {type} data...")

        # Initialize paths
        output_path = output_folder / type
        output_path.mkdir(exist_ok=True)

        # simply copy the metadata to the output folder
        metadata.to_csv(output_folder / f"metadata_{type}.csv", index=False)

        # Iterate over labels for preprocessing
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Processing {type} files"):
            audio_file = raw_data_path / f"{type}_submission" / row["FileName"]
            if not audio_file.exists():
                logger.error(f"File {audio_file} not found, skipping...")
                continue

            # Load the sound clip
            try:
                sample_rate, data = wavfile.read(audio_file)
                if len(data.shape) == 2:
                    data = data.mean(axis=1)  # Convert stereo to mono

                # Resample if necessary
                if sample_rate != SAMPLE_RATE:
                    data = librosa.resample(y=data.astype(float), orig_sr=sample_rate, target_sr=SAMPLE_RATE)

                # Handle audio length
                if len(data) < TARGET_SAMPLES:
                    # Pad with zeros if audio is too short
                    padding = TARGET_SAMPLES - len(data)
                    data = np.pad(data, (0, padding), mode="constant")
                elif len(data) > TARGET_SAMPLES:
                    # Randomly select a 10-second segment if audio is too long
                    max_start = len(data) - TARGET_SAMPLES
                    start = random.randint(0, max_start)
                    data = data[start : start + TARGET_SAMPLES]

                # Generate mel spectrogram
                S = librosa.feature.melspectrogram(y=data.astype(float), sr=sample_rate, n_mels=128, fmax=8000)
                S_DB = librosa.power_to_db(S, ref=np.max)

                # Normalize from [-80, 0] to [-1, 1] range
                S_DB = (S_DB + 40) / 40

                # Save processed audio features
                np.save(output_path / f"{audio_file.stem}.npy", S_DB)

            except Exception as e:
                logger.error(f"Error processing {audio_file}: {str(e)}")
                continue

    print("Preprocessing completed!")


if __name__ == "__main__":
    RAW_DATA = Path("data/raw")
    OUTPUT_FOLDER = Path("data/processed")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    typer.run(lambda: preprocess(raw_data_path=RAW_DATA, output_folder=OUTPUT_FOLDER, random_seed=42))
