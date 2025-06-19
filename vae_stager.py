import os

import torch
import torch.distributions as dist
import torch.nn as nn
import itertools
from tqdm import tqdm

from config import SIMULATED_OBSERVATIONS_DIR, SIMULATED_LABELS_DIR, DEVICE, MODEL_DIR
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
        return self.enc(X).mean

    def reconstruct(self, X):
        return self.dec(self.enc(X).sample()).sample()


def compute_elbo(vae, X, device):
    # q(z | X)
    q_z = vae.enc(X)

    # sample a vector z from q(z | X), using the reparameterisation trick
    epsilon = dist.Normal(0, 1).sample(q_z.mean.shape).to(device)  # not sure if this is right lol
    z = q_z.mean + epsilon * torch.sqrt(q_z.variance)

    # log prior : log p(z_i), chosen to be a standard multivariate gaussian
    log_prior = dist.Normal(0, 1).log_prob(z).sum(-1)

    # log posterior : log p(x_i | z_i)
    log_posterior = vae.dec(z).log_prob(X).sum(-1).sum(-1).sum(-1)

    # log q(z_i | x_i)
    log_qz = q_z.log_prob(z).sum(-1)

    return log_prior + log_posterior - log_qz


# if __name__ == "__main__":
#     dataset_name = "synthetic_120_10_dpm_0"
#     # VAE trying to infer stage with latent space
#     train_set = SyntheticDatasetVec(dataset_name=dataset_name)
#     train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
#
#     example_x, _ = next(iter(train_dataloader))
#     num_biomarkers = example_x.shape[1]
#
#     # Define network
#     enc = Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
#     dec = Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)
#
#     vae = VAE(enc=enc, dec=dec)
#
#     # Define optimiser
#     opt_vae = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=0.001)
#
#     # Run training loop
#     n_epochs = 50
#     train_vae(n_epochs, vae, train_dataloader, train_set, opt_vae, dataset_name, DEVICE)
#
#     # Evaluate by computing MSE error
#     print("Mean squared error:", evaluate_autoencoder(train_dataloader, vae, DEVICE)[0])

    # # VAE trying to infer seq with latent space
    # train_set = SyntheticDatasetMat(labels_dir=SIMULATED_LABELS_DIR, obs_dir=SIMULATED_OBSERVATIONS_DIR)
    # train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    # num_biomarkers = 10
    # d_obs = 1200
    #
    # enc = Encoder(d_in=d_obs, d_latent=num_biomarkers).to(DEVICE)
    # dec = Decoder(d_out=d_obs, d_latent=num_biomarkers).to(DEVICE)
    #
    # opt_vae = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()))
    #
    # # run training loop
    # n_epochs = 100
    # run_training(n_epochs, enc, dec, train_dataloader, train_set, opt_vae, DEVICE)
    #
    # # print label and latent of the training set after training
    # with torch.no_grad():
    #     for X, label in train_dataloader:
    #         X = X.to(DEVICE)
    #         X = X.reshape(-1, d_obs)
    #         print("ground truths: ", label)
    #
    #         latents = enc(X).sample()
    #         print("latents: ", latents)
    #
    #         # see if the latents correspond in some way to the ground truth orderings?
    #         print("latent ordering: ", torch.argsort(latents))
