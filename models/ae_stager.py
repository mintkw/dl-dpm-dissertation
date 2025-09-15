import os

import torch
import torch.nn as nn
import itertools
from tqdm import tqdm
from scipy import stats

from config import DEVICE, SAVED_MODEL_DIR
from datasets.biomarker_dataset import BiomarkerDataset
from dpm_algorithms.evaluation import evaluate_autoencoder
from models.autoencoder import AutoEncoder


class Encoder(nn.Module):

    def __init__(self, d_in, d_latent):
        super().__init__()
        self.d_latent = d_latent
        # self.net = nn.Sequential(nn.Linear(d_in, 32),
        #                          nn.ReLU(),
        #                          nn.Linear(32, 16),
        #                          nn.ReLU(),
        #                          nn.Linear(16, d_latent),
        #                          nn.Sigmoid())
        self.net = nn.Sequential(nn.Linear(d_in, 16),
                                 nn.ReLU(),
                                 nn.Linear(16, d_latent),
                                 nn.Sigmoid())
        self.latent_dir = nn.Parameter(torch.ones(1, requires_grad=False))  # 1 for ascending latent (0 is control and 1 is patient), 0 for descending

    def forward(self, X):
        h = self.net(X)

        return h


class Decoder(nn.Module):
    def __init__(self, d_out, d_latent):
        super().__init__()
        self.d_latent = d_latent
        # self.net = nn.Sequential(nn.Linear(self.d_latent, 16),
        #                          nn.ReLU(),
        #                          nn.Linear(16, 32),
        #                          nn.ReLU(),
        #                          nn.Linear(32, d_out))
        self.net = nn.Sequential(nn.Linear(self.d_latent, 16),
                                 nn.ReLU(),
                                 nn.Linear(16, d_out))

    def forward(self, Z):
        h = self.net(Z)

        return h


class AE(AutoEncoder):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec
        # self.latent_dir = 1  # 1 for ascending latent (0 is control and 1 is patient), 0 for descending

    def encode(self, X):
        return self.enc(X)

    def predict_uncorrected_stage(self, X):
        return self.enc(X)

    def predict_stage(self, X):
        uncorrected_stage = self.predict_uncorrected_stage(X)
        return self.enc.latent_dir * uncorrected_stage + (1 - self.enc.latent_dir) * (1 - uncorrected_stage)

    def decode_latent(self, z):
        return self.dec(z)


def ae_criterion(X, ae, device):
    reconstructions = ae.reconstruct_input(X)

    ms_error = ((X - reconstructions) ** 2).mean()

    return ms_error
