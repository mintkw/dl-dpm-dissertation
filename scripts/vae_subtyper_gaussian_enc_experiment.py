import os

import torch
import torch.distributions as dist
import torch.nn as nn
import itertools
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

from config import SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_OBS_TEST_DIR, SIMULATED_LABEL_TEST_DIR, DEVICE, SAVED_MODEL_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from evaluation import evaluate_autoencoder
from train_autoencoder import run_training
from plotting import predicted_stage_comparison
from models import vae_stager
from subtyping_utils import plot_dataset_in_latent_space, compute_subtype_accuracy


class Encoder(nn.Module):

    def __init__(self, n_biomarkers, d_latent, n_subtypes):
        super().__init__()
        self.d_latent = d_latent
        self.n_subtypes = n_subtypes
        self.n_biomarkers = n_biomarkers
        self.net = nn.Sequential(nn.Linear(self.n_biomarkers, 16),
                                 nn.ReLU())

        # self.fc_mu = nn.Linear(16, self.d_latent + self.n_subtypes)
        self.fc_stage_mu = nn.Linear(16, self.d_latent)
        self.fc_subtype_mu = nn.Linear(16, self.n_subtypes)

        self.fc_sigma = nn.Linear(16, self.d_latent + self.n_subtypes)

        self.latent_dir = nn.Parameter(torch.ones(1, requires_grad=False))  # 1 for ascending latent (0 is control and 1 is patient), 0 for descending

    def forward(self, X):
        h = self.net(X)
        stage_mu = torch.sigmoid(self.fc_stage_mu(h)) * self.n_biomarkers

        # The subtype output is meant to represent a normalised weightage over possible subtypes
        # subtype_mu = torch.sigmoid(self.fc_subtype_mu(h))
        temperature = 0.05
        subtype_output = self.fc_subtype_mu(h)
        subtype_mu = (torch.exp(subtype_output / temperature) /
                      torch.sum(torch.exp(subtype_output / temperature), dim=1).unsqueeze(1))

        log_sigma = self.fc_sigma(h)
        sigma = torch.exp(log_sigma)  # exponentiate to enforce non-negativity

        # Concatenate the computed stage and subtype means
        mu = torch.concatenate([stage_mu, subtype_mu], dim=-1)

        return dist.Normal(mu, sigma)


class Decoder(nn.Module):
    def __init__(self, d_out, d_latent, n_subtypes):
        super().__init__()
        self.d_latent = d_latent
        self.n_subtypes = n_subtypes
        self.net = nn.Sequential(nn.Linear(self.d_latent + self.n_subtypes, 16),
                                 nn.ReLU())

        self.fc_mu = nn.Linear(16, d_out)
        self.fc_sigma = nn.Linear(16, d_out)

    def forward(self, Z):
        h = self.net(Z)
        X_mu = self.fc_mu(h)
        X_sigma = torch.exp(self.fc_sigma(h))  # enforce non-negativity

        return dist.Normal(X_mu, X_sigma)


class VAE(vae_stager.VAE):
    def __init__(self, enc, dec):
        super().__init__(enc, dec)

    def predict_uncorrected_stage(self, X):
        return self.enc(X).mean[:, 0].unsqueeze(1) / X.shape[1]

    def predict_subtype(self, X):
        return torch.argmax(self.enc(X).mean[:, 1:], dim=-1)


def compute_elbo(vae, X, device, beta):
    # q(z | X)
    q_z = vae.enc(X)

    # sample a vector z from q(z | X), using the reparameterisation trick
    epsilon = dist.Normal(0, 1).sample(q_z.mean.shape).to(device)
    z = q_z.mean + epsilon * torch.sqrt(q_z.variance)

    # EXPERIMENT -------------------
    # calculate KL divergence from a sample
    # Let the prior be a mixture of Gaussians with uniform probability
    n_subtypes = z.shape[1] - 1
    pseudostage_prior = dist.Normal(0.5 * X.shape[1], 1 * X.shape[1])
    subtype_prior = dist.MixtureSameFamily(dist.Categorical(torch.ones(n_subtypes, device=DEVICE)),
                                           dist.Normal(torch.eye(n_subtypes, device=DEVICE), 0.5))
    kl_div = q_z.log_prob(z).sum(dim=-1) - pseudostage_prior.log_prob(z[:, 0]) - subtype_prior.log_prob(z[:, 1:]).sum(-1)
    kl_div = kl_div.squeeze(-1)
    # -----------------------------

    # negative expected reconstruction error
    posterior = vae.dec(z)
    expected_p_x = posterior.log_prob(X).sum(-1)

    # print(kl_div, expected_p_x)

    return -beta * kl_div + expected_p_x


def vae_criterion_wrapper(beta=1):
    def vae_criterion(X, vae, device):
        elbos = compute_elbo(vae, X, device, beta=beta)

        # The loss is the sum of the negative per-datapoint ELBO
        loss = -elbos.sum()

        return loss

    return vae_criterion


if __name__ == "__main__":
    num_sets = 3
    dataset_names = [f"synthetic_120_10_{i}" for i in range(num_sets)]
    model_name = "synthetic_120_10_mixed"

    train_set = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)

    val_set = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TEST_DIR, label_directory=SIMULATED_LABEL_TEST_DIR)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True)

    num_biomarkers = next(iter(train_loader))[0].shape[1]

    n_epochs = 1000

    # ---------- Train VAE -----------
    # Define network
    vae_enc = Encoder(n_biomarkers=num_biomarkers, d_latent=1, n_subtypes=num_sets).to(DEVICE)
    vae_dec = Decoder(d_out=num_biomarkers, d_latent=1, n_subtypes=num_sets).to(DEVICE)

    vae = VAE(enc=vae_enc, dec=vae_dec)

    # Define optimiser
    opt_vae = torch.optim.Adam(itertools.chain(vae_enc.parameters(), vae_dec.parameters()), lr=0.001)

    # Run training loop
    run_training(n_epochs, vae, model_name, train_loader, val_loader, opt_vae,
                 vae_criterion_wrapper(beta=1), model_type="vae", device=DEVICE)

    # EVALUATION ---------------------------------------------------------
    # Evaluate on the final models saved during training
    vae_enc_model_path = os.path.join(SAVED_MODEL_DIR, "vae", "enc_" + model_name + ".pth")
    vae_dec_model_path = os.path.join(SAVED_MODEL_DIR, "vae", "dec_" + model_name + ".pth")

    vae_enc.load_state_dict(torch.load(vae_enc_model_path, map_location=DEVICE))
    vae_dec.load_state_dict(torch.load(vae_dec_model_path, map_location=DEVICE))

    staging_mse, reconstruction_mse = evaluate_autoencoder(val_loader, vae, DEVICE)
    reconstruction_mse = reconstruction_mse.cpu()

    print("Staging RMSE of trained VAE on validation set:", staging_mse)
    print("Reconstruction MSE of trained VAE on validation set:", reconstruction_mse)

    # Compute subtype accuracy
    subtype_accuracy = compute_subtype_accuracy(vae, dataset_names)
    print("Subtype accuracy:", subtype_accuracy)

    # Visualise stage predictions against ground truth
    fig, ax = predicted_stage_comparison(train_loader, num_biomarkers, vae, DEVICE)
    fig.show()

    # Visualise the training set encoded in the latent space
    fig, ax = plot_dataset_in_latent_space(net=vae, dataset_names=dataset_names)

    fig.show()
    plt.show()
