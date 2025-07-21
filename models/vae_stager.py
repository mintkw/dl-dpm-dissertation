import os

import torch
import torch.distributions as dist
import torch.nn as nn
import itertools
from tqdm import tqdm
from scipy import stats

from config import SIMULATED_OBS_DIR, SIMULATED_LABEL_DIR, DEVICE, SAVED_MODEL_DIR
from datasets.synthetic_dataset_matrix import SyntheticDatasetMat
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from evaluation import evaluate_autoencoder


class Encoder(nn.Module):

    def __init__(self, d_in, d_latent):
        super().__init__()
        self.d_latent = d_latent

        self.net = nn.Sequential(nn.Linear(d_in, 16),
                                 nn.ReLU())

        self.fc_mu = nn.Linear(16, d_latent)
        self.fc_sigma = nn.Linear(16, d_latent)

        self.latent_dir = nn.Parameter(torch.ones(1, requires_grad=False))  # 1 for ascending latent (0 is control and 1 is patient), 0 for descending

    def forward(self, X):
        h = self.net(X)
        mu = torch.sigmoid(self.fc_mu(h))
        log_sigma = self.fc_sigma(h)
        sigma = torch.exp(log_sigma)  # exponentiate to enforce non-negativity

        return dist.Normal(mu, sigma)


class Decoder(nn.Module):
    def __init__(self, d_out, d_latent):
        super().__init__()
        self.d_latent = d_latent
        self.net = nn.Sequential(nn.Linear(self.d_latent, 16),
                                 nn.ReLU())

        self.fc_mu = nn.Linear(16, d_out)
        self.fc_sigma = nn.Linear(16, d_out)

    def forward(self, Z):
        h = self.net(Z)
        X_mu = self.fc_mu(h)
        X_sigma = torch.exp(self.fc_sigma(h))  # enforce non-negativity

        return dist.Normal(X_mu, X_sigma)


class VAE:
    def __init__(self, enc, dec):
        self.enc = enc
        self.dec = dec

    def encode(self, X):
        return self.enc.latent_dir * self.enc(X).mean + (1 - self.enc.latent_dir) * (1 - self.enc(X).mean)

    def decode(self, X):
        return self.dec(X).sample()  # or mean...?

    def reconstruct(self, X):
        return self.dec(self.enc(X).sample()).sample()

    def calculate_latent_direction(self, dataloader):
        batch_size = next(iter(dataloader))[0].shape[0]
        mean_correlation = 0.  # mean correlation across all variables and batches
        dataset_size = len(dataloader.dataset)

        for X, _ in dataloader:
            uncorrected_latents = self.enc(X).sample()

            # use mean correlation as a way to regularise the direction of the latent (lower for lower biomarker values)
            data_matrix = torch.concat([X, uncorrected_latents], dim=1)  # columns as variables and rows as observations

            correlation_matrix = stats.spearmanr(data_matrix.detach().cpu()).statistic
            mean_correlation += correlation_matrix[-1][:-1].mean() * batch_size / dataset_size

        if mean_correlation > 0:
            self.enc.latent_dir = nn.Parameter(torch.ones(1, requires_grad=False).to(DEVICE))
        else:
            self.enc.latent_dir = nn.Parameter(torch.zeros(1, requires_grad=False).to(DEVICE))


def compute_elbo(vae, X, device, beta=3):
    # q(z | X)
    q_z = vae.enc(X)

    # sample a vector z from q(z | X), using the reparameterisation trick
    epsilon = dist.Normal(0, 1).sample(q_z.mean.shape).to(device)  # not sure if this is right lol
    z = q_z.mean + epsilon * torch.sqrt(q_z.variance)

    # KL divergence between posterior approximation and prior, where the prior is a standard normal
    prior_mean = torch.zeros(1).to(device) + 0.5
    prior_variance = torch.ones(1).to(device)
    kl_div = 0.5 * (torch.log(prior_variance) - torch.log(q_z.variance) +
                    (q_z.variance + (q_z.mean - prior_mean) ** 2) / prior_variance - 1)

    kl_div = kl_div.sum(-1)  # per-datapoint ELBO is the sum over the per-latent ELBOs

    # # EXPERIMENT -------------------
    # # calculate KL divergence from a sample
    # prior = dist.Normal(0, 1)
    # kl_div = q_z.log_prob(z) - prior.log_prob(z)
    # kl_div = kl_div.squeeze(-1)
    # # -----------------------------

    # negative expected reconstruction error
    posterior = vae.dec(z)
    expected_p_x = posterior.log_prob(X).sum(-1)

    # print(kl_div, expected_p_x)

    # EXPERIMENT: BETA VAE -------------------------
    return -beta * kl_div + expected_p_x
    # ---------------------------------------------------

    return -kl_div + expected_p_x


def vae_criterion(X, vae, device):
    elbos = compute_elbo(vae, X, device)

    # The loss is the sum of the negative per-datapoint ELBO
    loss = -elbos.sum()

    return loss
