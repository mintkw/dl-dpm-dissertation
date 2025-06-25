# this class loads a dataset in which each data point is a dataset of measurements with
# their own underlying characteristic sequence, thus a matrix.
# labels are the sequence.

import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import json

from config import SIMULATED_LABEL_DIR, SIMULATED_OBS_DIR


class SyntheticDatasetMat(Dataset):
    def __init__(self, labels_dir, obs_dir):
        self.labels_dir = labels_dir
        self.obs_dir = obs_dir

        self.filenames = []  # list of file names holding observations as CSVs and labels as JSONs
        for file_path in os.listdir(self.obs_dir):
            filename = os.path.splitext(file_path)[0]
            if filename.split('_')[-1] == "kde-ebm":  # leave out kde-ebm format files
                continue
            self.filenames.append(filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # returns observations as a dataframe converted to a 2D tensor and labels as a 1D tensor
        instance_name = self.filenames[idx]
        obs_path = os.path.join(self.obs_dir, instance_name + ".csv")
        label_path = os.path.join(self.labels_dir, instance_name + "_seq.json")

        obs_df = pd.read_csv(obs_path)
        with open(label_path, 'r') as f:
            labels = json.load(f)

        observations = torch.tensor(obs_df.values, dtype=torch.float)[:, :-3]  # remove the last three columns which are CN, AD, MCI
        labels = torch.tensor(labels, dtype=torch.float).squeeze()

        return observations, labels

