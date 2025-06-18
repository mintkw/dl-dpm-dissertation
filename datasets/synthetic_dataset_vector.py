# this class loads a dataset in which each data point is a snapshot of measurements from one patient, thus a vector.
# labels are staging information.

import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import json

from config import SIMULATED_LABELS_DIR, SIMULATED_OBSERVATIONS_DIR


class SyntheticDatasetVec(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.obs_filepath = os.path.join(SIMULATED_OBSERVATIONS_DIR, dataset_name + ".csv")
        self.label_filepath = os.path.join(SIMULATED_LABELS_DIR, dataset_name + '_stages.json')

        self.obs_df = torch.tensor(pd.read_csv(self.obs_filepath).values, dtype=torch.float)[:, :-3]  # last 3 columns are CN, MCI, AD
        with open(self.label_filepath, 'r') as f:
            self.labels = json.load(f)


    def __len__(self):
        return len(self.obs_df)

    def __getitem__(self, idx):
        observations = self.obs_df[idx]
        labels = self.labels[idx]

        return observations, labels

