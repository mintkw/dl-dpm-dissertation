import os

import torch
import torch.nn as nn
import itertools
from tqdm import tqdm
from scipy import stats

from config import DEVICE, SAVED_MODEL_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from evaluation import evaluate_autoencoder
from models.autoencoder import AutoEncoder


class Encoder(nn.Module):

    def __init__(self, d_in, d_latent):
        super().__init__()
        self.d_latent = d_latent
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
        self.net = nn.Sequential(nn.Linear(self.d_latent, 16),
                                 nn.ReLU())

        self.fc = nn.Linear(16, d_out)

    def forward(self, Z):
        h = self.net(Z)

        return self.fc(h)


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

    # def automatically_set_latent_direction(self, dataloader):
    #     with torch.no_grad():
    #         batch_size = next(iter(dataloader))[0].shape[0]
    #         mean_correlation = 0.  # mean correlation across all variables and batches
    #         dataset_size = len(dataloader.dataset)
    #
    #         for X, _ in dataloader:
    #             uncorrected_latents = self.predict_uncorrected_stage(X)
    #
    #             # use mean correlation as a way to regularise the direction of the latent (lower for lower biomarker values)
    #             data_matrix = torch.concat([X, uncorrected_latents], dim=1)  # columns as variables and rows as observations
    #
    #             correlation_matrix = stats.spearmanr(data_matrix.detach().cpu()).statistic
    #             mean_correlation += correlation_matrix[-1][:-1].mean() * batch_size / dataset_size
    #
    #         if mean_correlation > 0:
    #             self.enc.latent_dir = nn.Parameter(torch.ones(1, requires_grad=False).to(DEVICE))
    #         else:
    #             self.enc.latent_dir = nn.Parameter(torch.zeros(1, requires_grad=False).to(DEVICE))


def ae_criterion(X, ae, device):
    reconstructions = ae.reconstruct_input(X)

    ms_error = ((X - reconstructions) ** 2).mean()

    return ms_error
