from torch.utils.data import Dataset

from src.instrument_classifier.data import InstrumentDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = InstrumentDataset()
    assert isinstance(dataset, Dataset)
