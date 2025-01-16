from pathlib import Path
import typer
from torch.utils.data import Dataset
import numpy as np
from scipy.io import wavfile
import librosa

# Create spectrogram with librosa
import matplotlib.pyplot as plt



class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    raw_data_path = Path(raw_data_path)
    for sound_clip in raw_data_path.iterdir():
        # Load the sound clip as a numpy array
        sample_rate, data = wavfile.read(sound_clip)
        if len(data.shape) == 2:
            data = data.mean(axis=1)
        print(sample_rate)
        continue


        return 
        # Generate the spectrogram
        S = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)

        # Save the spectrogram as an image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_DB, sr=sample_rate, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(output_folder / f"{sound_clip.stem}_spectrogram.png")
        plt.close()

        break
        # Normalize the data

        

    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
