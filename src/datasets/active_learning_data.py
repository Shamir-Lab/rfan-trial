import abc
import torch
import numpy as np

from torch.utils import data


class IActiveLearningDataset(abc.ABC):
    @abc.abstractmethod
    def acquire(self, pool_indices):
        pass

    @abc.abstractmethod
    def is_empty(self):
        pass


class ActiveLearningDataset(IActiveLearningDataset):
    # Originated from: https://github.com/BlackHC/batchbald_redux/blob/master/batchbald_redux/active_learning.py

    """ Splits `dataset` into an active dataset and an available pool dataset.
        Samples can be acquired from the pool to generate the active dataset.
        The initial dataset is assumed to be constant during the experiment (i.e, new samples are not added
        to the pool)

        This class can be used as follows:
        active_learning_data = ActiveLearningData(train_dataset)
        active_learning_data.acquire(initial_samples)
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

        # Before any acquisition, the pool data is full and training data is empty
        self.training_mask = np.full((len(dataset),), False)
        self.validation_mask = np.full((len(dataset),), False)
        self.pool_mask = np.full((len(dataset),), True)

        self.training_dataset = data.Subset(self.dataset, None)
        self.validation_dataset = data.Subset(self.dataset, None)
        self.pool_dataset = data.Subset(self.dataset, None)

        self._update_indices()

    def _update_indices(self):
        self.training_dataset.indices = np.nonzero(self.training_mask)[0]
        self.validation_dataset.indices = np.nonzero(self.validation_mask)[0]
        self.pool_dataset.indices = np.nonzero(self.pool_mask)[0]

    def _update_subsets(self):
        self.training_dataset = data.Subset(self.dataset, self.training_dataset.indices)
        self.validation_dataset = data.Subset(self.dataset, self.validation_dataset.indices)
        self.pool_dataset = data.Subset(self.dataset, self.pool_dataset.indices)

    @property
    def acquired_indices(self):
        """ return all acquiried indices, including both validation and training  """
        return np.concatenate((self.training_dataset.indices, self.validation_dataset.indices), axis=0)

    @property
    def training_indices(self):
        return self.training_dataset.indices

    @property
    def validation_indices(self):
        return self.validation_dataset.indices

    def is_empty(self):
        return len(self.pool_dataset) == 0

    def get_dataset_indices(self, pool_indices):
        """Transform indices (in `pool_dataset`) to indices in the original `dataset`."""
        indices = self.pool_dataset.indices[pool_indices]
        return indices

    def acquire(self, pool_indices, treatments, train_ratio=0.8):
        """Acquire elements from the pool dataset into the training dataset.
        Add them to training dataset & remove them from the pool dataset."""
        indices = self.get_dataset_indices(pool_indices)

        # Split to train and valid
        np.random.shuffle(indices)
        split_point = int(len(indices) * train_ratio)
        training_indices, validation_indices = indices[:split_point], indices[split_point:]

        # Update indices of patients to acquire
        self.training_mask[training_indices] = True
        self.validation_mask[validation_indices] = True
        self.pool_mask[indices] = False
        self._update_indices()

        # Enrol to trial and observe outcomes
        self.dataset.enrol(indices, treatments)

        # Update the actual content of the Subsets (with observed treatments and outcomes)
        self._update_subsets()

class RandomFixedLengthSampler(data.Sampler):
    """
    Custom data sampler that draws a fixed number of samples from a dataset with or without repetition.
    It is then used as a data loader sampler.

    [Sometimes, you really want to do more with little data without increasing the number of epochs.
    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition)].
    """
    def __init__(self, dataset, target_length):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < len(self.dataset):
            return iter(torch.randperm(len(self.dataset)).tolist())

        # Sample slightly more indices to avoid biasing towards start of dataset
        indices = torch.randperm(
            self.target_length + (-self.target_length % len(self.dataset))
        )

        return iter((indices[: self.target_length] % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length