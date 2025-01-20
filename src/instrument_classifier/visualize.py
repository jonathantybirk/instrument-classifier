import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from pathlib import Path
import sounddevice as sd
import pandas as pd
from tqdm import tqdm


def interactive_audio_viewer(
    processed_dir: str = "data/processed/train",
    raw_dir: str = "data/raw/train_submission",
    metadata_path: str = "data/raw/metadata_train.csv",
):
    """
    Interactive viewer for preprocessed spectrograms and their corresponding audio files.
    Use left/right arrow keys to navigate through files.
    Space to play/pause audio.
    Esc to exit.

    Args:
        processed_dir (str): Directory containing the preprocessed spectrograms
        raw_dir (str): Directory containing the raw audio files
        metadata_path (str): Path to the metadata CSV file
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Get list of preprocessed files
    processed_path = Path(processed_dir)
    raw_path = Path(raw_dir)

    # Get all numpy files (spectrograms)
    spectrogram_files = sorted(list(processed_path.glob("*.npy")))
    if not spectrogram_files:
        print(f"No .npy files found in {processed_dir}")
        return

    # Constants from preprocessing
    TARGET_DURATION = 10  # seconds
    SAMPLE_RATE = 44100  # Hz

    current_idx = 0
    playing = False
    current_audio = None
    current_sr = None
    current_cbar = None

    # Create figure and axes
    fig, (ax_spec, ax_wave) = plt.subplots(2, 1, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Spectrogram and Audio Viewer")

    # Initialize waveform plot
    (wave_line,) = ax_wave.plot([], [])

    def update_display(spec_file):
        nonlocal current_audio, current_sr, current_cbar

        # Load spectrogram
        S_db = np.load(spec_file)

        # Get corresponding audio file
        audio_file = raw_path / f"{spec_file.stem}.wav"

        # Get class from metadata
        file_info = metadata[metadata["FileName"] == f"{spec_file.stem}.wav"].iloc[0]
        instrument_class = file_info["Class"]

        # Load audio
        if audio_file.exists():
            current_audio, current_sr = librosa.load(str(audio_file))
        else:
            print(f"Warning: Audio file {audio_file} not found")
            current_audio, current_sr = None, None

        # Clear previous colorbar if it exists
        if current_cbar is not None:
            current_cbar.remove()

        # Clear and redraw spectrogram
        ax_spec.clear()
        img = librosa.display.specshow(
            S_db, x_axis="time", y_axis="mel", ax=ax_spec, sr=SAMPLE_RATE, hop_length=512
        )  # standard hop_length in librosa
        current_cbar = fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")
        ax_spec.set_title(f"File: {spec_file.stem} (Class: {instrument_class})")
        ax_spec.set_xlim(0, TARGET_DURATION)  # Set x-axis to show 10 seconds

        # Update waveform if audio exists
        if current_audio is not None:
            times = np.linspace(0, len(current_audio) / current_sr, len(current_audio))
            wave_line.set_data(times, current_audio)
            ax_wave.set_xlim(0, len(current_audio) / current_sr)
            ax_wave.set_ylim(current_audio.min(), current_audio.max())
        else:
            wave_line.set_data([], [])
            ax_wave.set_xlim(0, 1)
            ax_wave.set_ylim(-1, 1)

        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")

        # Refresh display
        plt.tight_layout()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def on_key(event):
        nonlocal current_idx, playing, current_audio, current_sr

        if event.key == "escape":
            plt.close(fig)

        elif event.key == "left":
            current_idx = (current_idx - 1) % len(spectrogram_files)
            if playing:
                sd.stop()
                playing = False
            update_display(spectrogram_files[current_idx])

        elif event.key == "right":
            current_idx = (current_idx + 1) % len(spectrogram_files)
            if playing:
                sd.stop()
                playing = False
            update_display(spectrogram_files[current_idx])

        elif event.key == " ":
            if not playing and current_audio is not None:
                sd.play(current_audio, current_sr)
                playing = True
            else:
                sd.stop()
                playing = False

    # Connect the key press event
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Load and display first file
    update_display(spectrogram_files[current_idx])

    print("\nControls:")
    print("Left/Right arrow keys: Navigate through files")
    print("Space: Play/pause audio")
    print("Esc: Exit viewer")

    plt.show()


def check_amplitude_stats(processed_dir: str = "data/processed/train"):
    """
    Check the maximum and minimum amplitudes of all processed spectrogram files in a directory.

    Args:
        processed_dir (str): Directory containing the processed spectrogram files (.npy)
    """
    processed_path = Path(processed_dir)
    spec_files = sorted(list(processed_path.glob("*.npy")))

    if not spec_files:
        print(f"No .npy files found in {processed_dir}")
        return

    max_amp = float("-inf")
    min_amp = float("inf")
    max_file = None
    min_file = None

    for spec_file in tqdm(spec_files, desc="Checking spectrogram stats"):
        spec_data = np.load(str(spec_file))
        current_max = spec_data.max()
        current_min = spec_data.min()

        if current_max > max_amp:
            max_amp = current_max
            max_file = spec_file.name
        if current_min < min_amp:
            min_amp = current_min
            min_file = spec_file.name

    print("\nSpectrogram Statistics:")
    print(f"Maximum value: {max_amp:.4f} dB (in file: {max_file})")
    print(f"Minimum value: {min_amp:.4f} dB (in file: {min_file})")


if __name__ == "__main__":
    # check_amplitude_stats()
    interactive_audio_viewer()
