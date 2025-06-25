# this class loads a dataset in which each data point is a snapshot of measurements from one patient, thus a vector.
# labels are staging information.

import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import json

from config import SIMULATED_LABEL_DIR, SIMULATED_OBS_DIR


class SyntheticDatasetVec(Dataset):
    def __init__(self, dataset_name, obs_directory, label_directory):
        self.dataset_name = dataset_name
        self.obs_filepath = os.path.join(obs_directory, dataset_name + ".csv")
        self.label_filepath = os.path.join(label_directory, dataset_name + '_stages.json')

        self.obs = torch.tensor(pd.read_csv(self.obs_filepath).values, dtype=torch.float)[:, :-3]  # last 3 columns are CN, MCI, AD
        with open(self.label_filepath, 'r') as f:
            self.labels = json.load(f)


    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        observations = self.obs[idx]
        labels = self.labels[idx]

        return observations, labels

