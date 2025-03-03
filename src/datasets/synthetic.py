import torch
import numpy as np

from torch.utils import data

from src.datasets import data_utils


class SyntheticDataGenerator(data.Dataset):
    """ Generates synthetic covariates x distributed Normal """
    def __init__(
        self,
        num_examples,
        mode,
        beta=0.75,
        sigma_y=1.0,
        bimodal=False,
        seed=1331,
        split=None,
    ):
        super(SyntheticDataGenerator, self).__init__()
        rng = np.random.RandomState(seed=seed)
        self.num_examples = num_examples
        self.dim_input = 1
        self.dim_treatment = 1
        self.dim_output = 1
        self.mode = mode
        if bimodal:
            self.x = np.vstack(
                [
                    rng.normal(loc=-2, scale=0.7, size=(num_examples // 2, 1)).astype(
                        "float32"
                    ),
                    rng.normal(loc=2, scale=0.7, size=(num_examples // 2, 1)).astype(
                        "float32"
                    ),
                ]
            )
        else:
            self.x = rng.normal(size=(num_examples, 1)).astype("float32")

        self.pi = (
            data_utils.complete_propensity(x=self.x, u=0, lambda_=1.0, beta=beta)
            .astype("float32")
            .ravel()
        )
        self.t = rng.binomial(1, self.pi).astype("float32")
        eps = (sigma_y * rng.normal(size=self.t.shape)).astype("float32")
        self.mu0 = data_utils.f_mu(x=self.x, t=0.0, u=0, gamma=0.0).astype("float32").ravel()
        self.mu1 = data_utils.f_mu(x=self.x, t=1.0, u=0, gamma=0.0).astype("float32").ravel()
        self.y0 = self.mu0 + eps
        self.y1 = self.mu1 + eps
        self.tau = self.mu1 - self.mu0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index : index + 1]


    def get_sensitive_subgroups(self, ranges):
        """
        Args: ranges (list of str): List of range tuples representing sensitive groups [(0, 12)].

        Returns: List of binary numpy.ndarray indicating subgroups of self.x according to ranges.
        """
        subgroups = {}
        for range in ranges:
            min_value, max_value = range[0], range[1]
            subgroup = (self.x >= min_value) & (self.x < max_value)
            subgroups[str(range)] = subgroup.ravel()

        return subgroups


class SyntheticPool(SyntheticDataGenerator):
    """ Inherent from SyntheticDataGenerator and initializes empty treatments and outcomes, as before the trial
        there's no access to treatment arms and outcomes """
    def __init__(self, num_examples, mode, beta=0.75, sigma_y=1.0, bimodal=False, seed=1330):
        super().__init__(num_examples, mode, beta, sigma_y, bimodal, seed)

        # Init with Nans - Before the trial there's no access to treatment arms and outcomes
        self.t = np.full(self.num_examples, np.nan)
        self.y = np.full(self.y1.shape, np.nan)
        if mode == "pi":
            self.inputs = self.x
            self.targets = self.t
        elif mode == "mu":
            self.inputs = np.hstack([self.x, np.expand_dims(self.t, -1)])
            self.targets = self.y
        else:
            raise NotImplementedError(
                f"{mode} not supported. Choose from 'pi'  for propensity models or 'mu' for expected outcome models"
            )
        self.y_mean = np.array([0.0], dtype="float32")
        self.y_std = np.array([1.0], dtype="float32")
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def set_t(self, indices, t):
        self.t[indices] = t[indices]
        if self.mode == "pi":
            self.targets[indices] = torch.tensor(self.t[indices], dtype=torch.float32)
        elif self.mode == "mu":
            updated_inputs = np.hstack([self.x, np.expand_dims(self.t, -1)])[indices]
            self.inputs[indices] = torch.tensor(updated_inputs, dtype=torch.float32)

    def set_y_obs(self, indices):
        self.y[indices] = self.t[indices] * self.y1[indices] + (1 - self.t[indices]) * self.y0[indices]
        if self.mode == "mu":
            self.targets[indices] = torch.tensor(self.y[indices], dtype=torch.float32)

    def enrol(self, indices, t):
        print(f"Enrolling {len(indices)} patients")
        self.set_t(indices, t)
        self.set_y_obs(indices)


class SyntheticVal(SyntheticDataGenerator):
    """ Inherent from SyntheticDataGenerator and initializes t """
    def __init__(self, num_examples, mode, beta=0.75, sigma_y=1.0, bimodal=False, seed=1331):
        super().__init__(num_examples, mode, beta, sigma_y, bimodal, seed)
        self.y = self.t * self.y1 + (1 - self.t) * self.y0
        if mode == "pi":
            self.inputs = self.x
            self.targets = self.t
        elif mode == "mu":
            self.inputs = np.hstack([self.x, np.expand_dims(self.t, -1)])
            self.targets = self.y
        else:
            raise NotImplementedError(
                f"{mode} not supported. Choose from 'pi'  for propensity models or 'mu' for expected outcome models"
            )
        self.y_mean = np.array([0.0], dtype="float32")
        self.y_std = np.array([1.0], dtype="float32")
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
