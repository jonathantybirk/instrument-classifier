from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from unittest.mock import patch
from instrument_classifier.data import preprocess
from instrument_classifier.data import InstrumentDataset


def test_my_dataset():
    """Test the InstrumentDataset class."""
    dataset = InstrumentDataset(Path("data/processed/train"), Path("data/processed/metadata_train.csv"))
    assert isinstance(dataset, Dataset)

    # Test a single item from the dataset
    item = dataset[0]
    assert isinstance(item, tuple), "Dataset item should be a tuple"
    assert len(item) == 2, "Dataset item should contain (features, label)"

    features, label = item
    assert isinstance(features, np.ndarray), "Features should be a numpy array"
    assert isinstance(label, int), "Label should be an integer"

    # Test dataset length
    assert len(dataset) > 0, "Dataset should not be empty"

    # Test multiple items
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        features, label = item
        assert features.shape[0] > 0, f"Features at index {i} should not be empty"
        assert 0 <= label < 10, f"Label at index {i} should be between 0 and 9"


def test_preprocess_without_saving(tmp_path):
    """Test the preprocess function without actually saving files."""
    # Create a temporary directory structure
    raw_data_path = tmp_path / "raw"
    raw_data_path.mkdir()
    (raw_data_path / "train_submission").mkdir()

    # Create a test audio file (10 seconds of silence at 44.1kHz)
    test_audio = np.zeros(441000)  # 10 seconds of silence

    # Create a minimal metadata CSV
    metadata = pd.DataFrame({"FileName": ["test.wav"], "Class": ["piano"]})
    metadata.to_csv(raw_data_path / "metadata_train.csv", index=False)

    # Mock saving the audio file
    with (
        patch("scipy.io.wavfile.read") as mock_read,
        patch("numpy.save") as mock_save,
        patch("pandas.read_csv", return_value=metadata),
        patch("pathlib.Path.exists", return_value=True),
    ):
        # Configure mock to return our test audio data
        mock_read.return_value = (44100, test_audio)

        # Run preprocessing
        output_folder = tmp_path / "processed"
        preprocess(raw_data_path, output_folder)

        # Verify that numpy.save was called
        assert mock_save.called, "numpy.save should have been called"

        # Get the spectrogram that would have been saved
        saved_data = mock_save.call_args[0][1]

        # Verify the exact shape and properties of the processed data
        assert isinstance(saved_data, np.ndarray), "Processed data should be a numpy array"
        assert saved_data.shape == (128, 862), f"Expected shape (128, 862), got {saved_data.shape}"
        assert -1 <= saved_data.min() <= saved_data.max() <= 1, "Data should be normalized between -1 and 1"
