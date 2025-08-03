# Parent abstract class from which AE and VAE are derived.

import torch
import abc
from scipy import stats
from torch import nn
from config import DEVICE


class AutoEncoder(abc.ABC):
    def __init__(self):
        self.enc = None
        self.dec = None

    @abc.abstractmethod
    def encode(self, X):
        """
        Args:
            X: Input

        Returns:
            The output of the network's encoder.
        """
        return self.enc(X)

    @abc.abstractmethod
    def predict_stage(self, X):
        pass

    @abc.abstractmethod
    def predict_uncorrected_stage(self, X):
        pass

    @abc.abstractmethod
    def decode_latent(self, z):
        pass

    def reconstruct_input(self, X):
        return self.decode_latent(self.encode(X))

    def automatically_set_latent_direction(self, dataloader):
        with torch.no_grad():
            batch_size = next(iter(dataloader))[0].shape[0]
            mean_correlation = 0.  # mean correlation across all variables and batches
            dataset_size = len(dataloader.dataset)

            for X, _ in dataloader:
                uncorrected_latents = self.predict_uncorrected_stage(X)

                # use mean correlation as a way to regularise the direction of the latent (lower for lower biomarker values)
                data_matrix = torch.concat([X, uncorrected_latents], dim=1)  # columns as variables and rows as observations

                correlation_matrix = stats.spearmanr(data_matrix.detach().cpu()).statistic
                mean_correlation += correlation_matrix[-1][:-1].mean() * batch_size / dataset_size

            if mean_correlation > 0:
                self.enc.latent_dir = nn.Parameter(torch.ones(1, requires_grad=False).to(DEVICE))
            else:
                self.enc.latent_dir = nn.Parameter(torch.zeros(1, requires_grad=False).to(DEVICE))