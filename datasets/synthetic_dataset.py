# This class loads a dataset in which each data point is a snapshot of measurements from one patient, thus a vector.
# labels are staging information.

import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import json

from config import DEVICE


class SyntheticDataset(Dataset):
    def __init__(self, dataset_names, obs_directory, label_directory=None):
        """
        Args:
            dataset_names (str or list[str]): A list of dataset names. Expects datasets to have the same dimensions.
            obs_directory: Path to directory containing observation data
            label_directory (optional): Path to directory containing label data
        """
        if type(dataset_names) is not list:
            dataset_names = [dataset_names]

        self.dataset_names = dataset_names

        self.obs = []
        self.labels = []
        self.biomarker_names = []
        for dataset_name in dataset_names:
            obs_filepath = os.path.join(obs_directory, dataset_name + ".csv")

            # Note: last 3 columns are CN, MCI, AD
            df = pd.read_csv(obs_filepath)
            self.obs.append(torch.tensor(df.values, dtype=torch.float))
            self.biomarker_names = df.columns[:-3]

            if label_directory is not None:
                label_filepath = os.path.join(label_directory, dataset_name + '_stages.json')

                with open(label_filepath, 'r') as f:
                    self.labels.append(torch.tensor(json.load(f)))

        self.obs = torch.concatenate(self.obs, dim=0).to(DEVICE)

        if label_directory is not None:
            self.labels = torch.concatenate(self.labels, dim=0).to(DEVICE)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        observations = self.obs[idx][:-3]

        if len(self.labels) > 0:
            label = self.labels[idx]
        else:
            label = observations.shape[0] * torch.argmax(self.obs[idx][-3:]) / 2  # simply encode CN, MCI, AD as the label

        return observations, label

