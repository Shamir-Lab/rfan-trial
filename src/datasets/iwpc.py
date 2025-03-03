import torch
import numpy as np
import pandas as pd
from torch.utils import data

from sklearn import preprocessing
from sklearn import model_selection

CONTINUOUS_COVARS = ['age_group', 'height', 'weight', 'bmi']

BINARY_COVARS = [# Demographics
                'male', 'white', 'asian', 'black', 'non_hispanic', 'no_age',
                # Treatment cause
                'indication for warfarin treatment: 1',
                'indication for warfarin treatment: 2',
                'indication for warfarin treatment: 3',
                'indication for warfarin treatment: 4',
                'indication for warfarin treatment: 5',
                'indication for warfarin treatment: 6',
                'indication for warfarin treatment: 7',
                'indication for warfarin treatment: 8',
                # Comorbidities
                'diabetes_0', 'diabetes_1',
                'CHF_cardiomyopathy_0', 'CHF_cardiomyopathy_1',
                'valve_replacement_0', 'valve_replacement_1',
                # Medications
                'Aspirin_0', 'Aspirin_1',
                'Acetaminophen or Paracetamol (Tylenol)_0', 'Acetaminophen or Paracetamol (Tylenol)_1',
                'Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day_0',
                'Simvastatin (Zocor)_0', 'Simvastatin (Zocor)_1', 'Atorvastatin (Lipitor)_0',
                'Atorvastatin (Lipitor)_1', 'Fluvastatin (Lescol)_0', 'Lovastatin (Mevacor)_0',
                'Lovastatin (Mevacor)_1', 'Pravastatin (Pravachol)_0', 'Pravastatin (Pravachol)_1',
                'Rosuvastatin (Crestor)_0', 'Rosuvastatin (Crestor)_1',
                'Cerivastatin (Baycol)_0', 'Amiodarone (Cordarone)_0',
                'Amiodarone (Cordarone)_1', 'Carbamazepine (Tegretol)_0',
                'Carbamazepine (Tegretol)_1', 'Phenytoin (Dilantin)_0',
                'Phenytoin (Dilantin)_1', 'Rifampin or Rifampicin_0',
                'Sulfonamide Antibiotics_0', 'Macrolide Antibiotics_0',
                'Anti-fungal Azoles_0', 'Anti-fungal Azoles_1',
                'Herbal Medications, Vitamins, Supplements_0',
                'Herbal Medications, Vitamins, Supplements_1',
                # Current smoker
                'smoker_0', 'smoker_1',
                # Cyp2C9 genotypes
                'CYP2C9_*1/*1', 'CYP2C9_*1/*2', 'CYP2C9_*1/*3', 'CYP2C9_NA',
                # VKORC1 genotypes
                'VKORC1 -1639 A/G', 'VKORC1 -1639 A/A', 'VKORC1 -1639 G/G',
                'VKORC1 497 G/T', 'VKORC1 497 T/T', 'VKORC1 497 G/G',
                'VKORC1 1173 T/T', 'VKORC1 1173 C/T', 'VKORC1 1173 C/C',
                'VKORC1 1542 C/G', 'VKORC1 1542 C/C', 'VKORC1 1542 G/G',
                'VKORC1 3730 A/G', 'VKORC1 3730 G/G', 'VKORC1 3730 A/A',
                'VKORC1 2255 T/T', 'VKORC1 2255 C/T', 'VKORC1 2255 C/C',
                'VKORC1 -4451 C/C', 'VKORC1 -4451 A/C', 'VKORC1 -4451 A/A']


class IWPC(data.Dataset):
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
            self.unscaled_age = df_test['age_group'].to_numpy(dtype="float32")
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
            self.unscaled_age = df['age_group'].to_numpy(dtype="float32")
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
        
        # Define ndarray masks by sensitive groups: gender, race.
        
        mask = {}
        
        # Gender
        idx = self.columns.get_loc('male')
        mask["male"] = (self.x[:, idx] == 1)
        mask["female"] = (self.x[:, idx] == 0)
    
        # Race
        for race in ["white", "asian", "black"]:
            idx = self.columns.get_loc(race)
            mask[race] = (self.x[:, idx] == 1)
        
        subgroups = {}
        for gender in ["male", "female"]:
            for race in ["white", "asian", "black"]:
                subgroups[f"{gender}_{race}"] = mask[gender] & mask[race]

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
