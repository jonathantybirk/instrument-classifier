from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

from src.instrument_classifier.data import InstrumentDataset


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
