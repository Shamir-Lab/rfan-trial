from src.datasets.synthetic import SyntheticPool, SyntheticVal
from src.datasets.iwpc import IWPC
from src.datasets.covid import Covid

from src.datasets.active_learning_data import ActiveLearningDataset
from src.datasets.active_learning_data import RandomFixedLengthSampler

DATASETS = {
    "synthetic": SyntheticPool,
    "synthetic_valid": SyntheticVal,
    "iwpc": IWPC,
    "covid": Covid,
}
