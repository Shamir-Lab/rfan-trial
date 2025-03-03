import torch
import numpy as np
import pandas as pd
from torch.utils import data

from sklearn import preprocessing
from sklearn import model_selection

CONTINUOUS_COVARS = ['Age']

BINARY_COVARS = ['Sex_male',
                 # Symptoms
                 'Fever', 'Cough', 'Sore_throat', 'Shortness_of_breath', 'Respiratory_discomfort',
                 'SPO2', 'Dihareea', 'Vomitting',
                 # Comorbidities
                 'Cardiovascular', 'Asthma', 'Diabetis', 'Pulmonary', 'Immunosuppresion',
                 'Obesity', 'Liver', 'Neurologic', 'Renal',
                 # Race
                 'Branca', 'Preta', 'Amarela', 'Parda', 'Indigena',
                 # Areas
                 'Northeast', 'North', 'Southeast', 'Central-West', 'South']


class Covid(data.Dataset):
    def __init__(self, data_path, split, seed):
        # Load parsed data
        df = pd.read_csv(data_path, index_col=0)
        assert df[CONTINUOUS_COVARS + BINARY_COVARS + ["y1", "y0"]].shape == df.shape

        # Set MUs
        df["mu0"] = df["y0"]
        df["mu1"] = df["y1"]

        # Train test split
        df_train, df_test = model_selection.train_test_split(df, test_size=0.2, random_state=seed)

        self.split = split

        # Set x, y, and t values
        covars = CONTINUOUS_COVARS + BINARY_COVARS

        self.columns = df[covars].columns
        self.dim_input = len(covars)
        self.dim_treatment = 1
        self.dim_output = 1
        
        if self.split == "test":
            # Standardize continuous covariates
            df_test[CONTINUOUS_COVARS] = preprocessing.StandardScaler().fit_transform(df_test[CONTINUOUS_COVARS])

            self.x = df_test[covars].to_numpy(dtype="float32")
            self.mu0 = df_test["mu0"].to_numpy(dtype="float32")
            self.mu1 = df_test["mu1"].to_numpy(dtype="float32")
            self.y0 = df_test["y0"].to_numpy(dtype="float32")
            self.y1 = df_test["y1"].to_numpy(dtype="float32")
            self.tau = self.mu1 - self.mu0
        else:
            if split == "train":
                df = df_train
            else:
                raise NotImplementedError("Not a valid dataset split")

            # Standardize continuous covariates
            df[CONTINUOUS_COVARS] = preprocessing.StandardScaler().fit_transform(df[CONTINUOUS_COVARS])

            self.x = df[covars].to_numpy(dtype="float32")
            self.mu0 = df["mu0"].to_numpy(dtype="float32")
            self.mu1 = df["mu1"].to_numpy(dtype="float32")
            self.y0 = df["y0"].to_numpy(dtype="float32")
            self.y1 = df["y1"].to_numpy(dtype="float32")
        
        # Unkown treatment and outcome before trial
        self.t = np.full(self.y1.shape, np.nan)
        self.y = np.full(self.y1.shape, np.nan)
        
        self.inputs = np.hstack([self.x, np.expand_dims(self.t, -1)])
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = self.y
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        print(self.x.shape)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index : index + 1]

    def get_sensitive_subgroups(self, ranges=None):
        """
        Returns: List of binary numpy.ndarray indicating subgroups of self.x according to ranges.
        """

        subgroups = {}

        # Race
        ethnic_groups = ['Branca', 'Preta', 'Amarela', 'Parda', 'Indigena']
        for race in ethnic_groups:
            idx = self.columns.get_loc(race)
            subgroups[race] = (self.x[:, idx] == 1)
            
        # Area
        areas = ['Northeast', 'North', 'Southeast', 'Central-West', 'South']
        for area in areas:
            idx = self.columns.get_loc(area)
            subgroups[area] = (self.x[:, idx] == 1)
                    
        return subgroups

    
    def set_t(self, indices, t):
        self.t[indices] = t[indices]
        updated_inputs = np.hstack([self.x, np.expand_dims(self.t, -1)])[indices]
        self.inputs[indices] = torch.tensor(updated_inputs, dtype=torch.float32)

    def set_y_obs(self, indices):
        self.y[indices] = self.t[indices] * self.y1[indices] + (1 - self.t[indices]) * self.y0[indices]
        self.targets[indices] = torch.tensor(self.y[indices], dtype=torch.float32)

    def enrol(self, indices, t):
        print(f"Enrolling {len(indices)} patients")
        self.set_t(indices, t)
        self.set_y_obs(indices)
