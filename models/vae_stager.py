import os

import torch
import torch.distributions as dist
import torch.nn as nn
import itertools
from tqdm import tqdm
from scipy import stats

from config import SIMULATED_OBS_DIR, SIMULATED_LABEL_DIR, DEVICE, SAVED_MODEL_DIR
from datasets.synthetic_dataset import SyntheticDataset
from dpm_algorithms.evaluation import evaluate_autoencoder
from models.autoencoder import AutoEncoder


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


class VAE(AutoEncoder):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def encode(self, X):
        return self.enc(X).sample()

    def predict_stage(self, X):
        uncorrected_stage = self.predict_uncorrected_stage(X)
        return self.enc.latent_dir * uncorrected_stage + (1 - self.enc.latent_dir) * (1 - uncorrected_stage)

    def predict_uncorrected_stage(self, X):
        return self.enc(X).mean

    def decode_latent(self, z):
        return self.dec(z).mean


def compute_elbo(vae, X, device, beta):
    # q(z | X)
    q_z = vae.enc(X)

    # sample a vector z from q(z | X), using the reparameterisation trick
    epsilon = dist.Normal(0, 1).sample(q_z.mean.shape).to(device)  # not sure if this is right lol
    z = q_z.mean + epsilon * torch.sqrt(q_z.variance)

    # KL divergence between posterior approximation and prior, analytically computed since both are Gaussian
    prior_mean = torch.zeros(1).to(device) + 0.5
    prior_variance = torch.ones(1).to(device)
    kl_div = 0.5 * (torch.log(prior_variance) - torch.log(q_z.variance) +
                    (q_z.variance + (q_z.mean - prior_mean) ** 2) / prior_variance - 1)

    kl_div = kl_div.sum(-1)  # per-datapoint ELBO is the sum over the per-latent ELBOs

    # negative expected reconstruction error
    posterior = vae.dec(z)
    expected_p_x = posterior.log_prob(X).sum(-1)

    return -beta * kl_div + expected_p_x


def vae_criterion_wrapper(beta=1):
    def vae_criterion(X, vae, device):
        elbos = compute_elbo(vae, X, device, beta=beta)

        # The loss is the sum of the negative per-datapoint ELBO
        loss = -elbos.sum()

        return loss

    return vae_criterion
