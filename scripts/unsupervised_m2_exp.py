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
# from train_autoencoder import run_training
from plotting import predicted_stage_comparison
from models import vae_stager
from subtyping_utils import plot_dataset_in_latent_space, compute_subtype_accuracy_with_cluster_mapping


class Encoder(nn.Module):

    def __init__(self, n_biomarkers, d_latent, n_subtypes):
        super().__init__()
        # self.d_latent = d_latent
        self.n_subtypes = n_subtypes
        self.n_biomarkers = n_biomarkers

        self.net = nn.Sequential(nn.Linear(self.n_biomarkers, 16),
                                 nn.ReLU())

        # self.fc_mu = nn.Linear(16, self.n_subtypes)  # mu_phi(x, y). one output for each y class (subtype)
        self.fc_mu = nn.Linear(16, 1)
        self.fc_log_var = nn.Linear(16, 1)  # sigma^2_phi(x)
        self.fc_pi = nn.Linear(16, self.n_subtypes)  # pi_phi(x)

        self.latent_dir = nn.Parameter(torch.ones(1, requires_grad=False))  # 1 for ascending latent (0 is control and 1 is patient), 0 for descending

    def forward(self, X):
        h = self.net(X)

        mu = torch.sigmoid(self.fc_mu(h))
        var = torch.exp(self.fc_log_var(h))
        pi = torch.nn.functional.softmax(self.fc_pi(h), dim=-1)

        return mu, var, pi


class Decoder(nn.Module):
    def __init__(self, d_out, d_z, n_subtypes):
        super().__init__()
        self.d_z = d_z  # dimensionality of z (stage), expected to be 1
        self.d_out = d_out
        self.n_subtypes = n_subtypes

        self.net = nn.Sequential(nn.Linear(self.d_z + self.n_subtypes, 16),
                                 nn.ReLU())

        self.fc_mu = nn.Linear(16, self.d_out)  # mu_theta(z, y)
        self.fc_log_var = nn.Linear(16, self.d_out)  # sigma^2_theta(z, y)

    def forward(self, Z):
        # Expects Z to be z, y concatenated in that order.

        h = self.net(Z)
        mu = self.fc_mu(h)
        var = torch.exp(self.fc_log_var(h))  # enforce non-negativity

        return mu, var


class VAE(vae_stager.VAE):
    def __init__(self, enc, dec):
        super().__init__(enc, dec)

    def encode(self, X):
        mu, var, pi = self.enc(X)
        y_idx_sample = dist.Categorical(pi).sample()  # shape (batch_size,)
        # mu = mu[torch.arange(mu.shape[0]), y_idx_sample]  # select each mean based on the sampled y
        z_sample = dist.Normal(mu, var).sample()  # shape (batch_size,)

        # Expand y into a one-hot vector
        y_sample = torch.zeros(pi.shape, device=DEVICE)
        y_sample[torch.arange(pi.shape[0]), y_idx_sample] = 1.

        return torch.concatenate([z_sample, y_sample], dim=-1)

    def predict_uncorrected_stage(self, X):
        mu, var, pi = self.enc(X)
        y = torch.argmax(pi, dim=-1)  # shape (batch_size,)

        return mu

        # stages = mu[torch.arange(X.shape[0]), y]

        # return stages.unsqueeze(1)

    def predict_stage(self, X):
        uncorrected_stage = self.predict_uncorrected_stage(X)
        return self.enc.latent_dir * uncorrected_stage + (1 - self.enc.latent_dir) * (1 - uncorrected_stage)

    def decode_latent(self, z):
        mu, var = self.dec(z)

        return dist.Normal(mu, var).sample()

    def predict_subtype(self, X):
        mu, var, pi = self.enc(X)
        y = torch.argmax(pi, dim=-1)  # shape (batch_size,)

        return y

    def subtype_scores(self, X):
        # Assign a (normalised) score of how likely X is to belong to each subtype
        _, _, q_pi = self.enc(X)

        return q_pi


def compute_elbo(vae, X, device, beta):
    # q(z | X)
    q_mu, q_var, q_pi = vae.enc(X)

    n_subtypes = q_pi.shape[1]
    batch_size = X.shape[0]

    # y = torch.arange(n_subtypes, device=DEVICE)  # shape: (n_subtypes,)
    y = torch.eye(n_subtypes, device=DEVICE)

    # term 1: Expectation over q_phi(z|x,y) of log p_theta(x|y,z). Calculated with the reparametrisation trick
    epsilon = dist.Normal(0, 1).sample(q_mu.shape).to(device)

    z = q_mu + epsilon * torch.sqrt(q_var)  # transformed from epsilon (batch_size, 1)
    decoder_input = torch.concatenate([z.repeat(1, n_subtypes).unsqueeze(-1), y.repeat((batch_size, 1, 1))],
                                      dim=-1)  # (batch_size, n_subtypes, 1 + n_subtypes)
    p_mu, p_var = vae.dec(decoder_input)
    posterior_over_x = dist.Normal(p_mu, p_var)
    log_px_given_yz = posterior_over_x.log_prob(X.unsqueeze(1).repeat(1, n_subtypes, 1)).sum(
        -1)  # expected shape: (batch_size, n_subtypes, 1)

    # z = q_mu + epsilon * torch.sqrt(q_var)  # transformed from epsilon (batch_size, n_subtypes)
    # decoder_input = torch.concatenate([z.unsqueeze(-1), y.repeat((batch_size, 1, 1))], dim=-1)  # (batch_size, n_subtypes, 1 + n_subtypes)
    # p_mu, p_var = vae.dec(decoder_input)
    # posterior_over_x = dist.Normal(p_mu, p_var)
    # log_px_given_yz = posterior_over_x.log_prob(X.unsqueeze(1).repeat(1, n_subtypes, 1)).sum(-1)  # expected shape: (batch_size, n_subtypes, 1)

    # term 2: log p(y). The prior is uniform.
    log_py = torch.log(torch.ones(X.shape[0], n_subtypes, device=DEVICE) / n_subtypes)  # expected shape: (batch_size, n_subtypes)

    # term 3: D_KL[q_phi(z|x,y) || p(z)]
    prior_mean = torch.zeros(1).to(device) + 0.5
    prior_variance = torch.ones(1).to(device)
    kl_div = 0.5 * (torch.log(prior_variance) - torch.log(q_var) +
                    (q_var + (q_mu - prior_mean) ** 2) / prior_variance - 1)  # expected shape: (batch_size, n_subtypes)
    kl_div = kl_div.repeat(1, n_subtypes)

    # term 4: log q_phi(y | x)
    log_qy_given_x = torch.log(q_pi)  # expected shape: (batch_size, n_subtypes)

    # Take the expected value over all y
    # print("log p(x|y,z):", log_px_given_yz)
    # print("log p(y)", log_py)
    # print("KL div", kl_div)
    # print("log q(y|x)", log_qy_given_x)
    elbo = q_pi * (log_px_given_yz + log_py - kl_div - log_qy_given_x)
    elbo = elbo.sum(-1)

    return elbo  # expected shape: (batch_size, 1)


def vae_criterion_wrapper(beta=1):
    def vae_criterion(X, vae, device):
        elbos = compute_elbo(vae, X, device, beta=beta)

        # The loss is the sum of the negative per-datapoint ELBO
        loss = -elbos.sum()

        return loss

    return vae_criterion


def run_training(n_epochs, net, model_name, train_loader, val_loader, optimiser, criterion, model_type, device):
    train_dataset_size = len(train_loader.dataset)
    model_dir = os.path.join(SAVED_MODEL_DIR, model_type)
    os.makedirs(model_dir, exist_ok=True)

    enc_path = os.path.join(model_dir, "enc_" + model_name + ".pth")
    dec_path = os.path.join(model_dir, "dec_" + model_name + ".pth")

    epochs_without_improvement = 0
    best_loss = float('inf')
    epoch_patience = 10
    minimum_improvement = 1e-4  # Minimum improvement considered 'significant'

    for epoch in tqdm(range(n_epochs), desc=f"Training {model_type}"):
        train_loss = 0.0
        for (X, _) in train_loader:
            X = X.to(device)
            optimiser.zero_grad()

            loss = criterion(X, net, device)
            loss.backward()

            optimiser.step()
            train_loss += loss.item() * X.shape[0] / train_dataset_size

        # Compute loss on validation set
        val_loss = 0.0
        val_dataset_size = len(val_loader.dataset)
        with torch.no_grad():
            for (X, _) in val_loader:
                X = X.to(device)
                optimiser.zero_grad()

                val_loss += criterion(X, net, device).item() * X.shape[0] / val_dataset_size

        # Compute error between latents and stages, just to track progress
        # rmse_stage_error, reconstruction_error = evaluate_autoencoder(train_loader, net, device)

        if best_loss - val_loss >= minimum_improvement:
            best_loss = val_loss
            torch.save(net.enc.state_dict(), enc_path)
            torch.save(net.dec.state_dict(), dec_path)
            epochs_without_improvement = 0
        else:
            # Terminate training early if no significant improvement
            epochs_without_improvement += 1

            if epochs_without_improvement >= epoch_patience:
                print(
                    f"Ending training early as no significant validation loss decrease "
                    f"has been observed in {epoch_patience} epochs")
                break

        if epoch % 10 == 0:
            subtype_accuracy = compute_subtype_accuracy_with_cluster_mapping(net, dataset_names)  # todo: just directly uses the one from outside
            print(
                f"Epoch {epoch}, train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, subtype accuracy = {subtype_accuracy:.4f}")

    # Load the best model and call compute_latent_direction then store it again before returning
    net.enc.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    net.dec.load_state_dict(torch.load(dec_path, map_location=DEVICE))

    net.automatically_set_latent_direction(train_loader)

    torch.save(net.enc.state_dict(), enc_path)
    torch.save(net.dec.state_dict(), dec_path)


if __name__ == "__main__":
    num_sets = 2
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
    vae_dec = Decoder(d_out=num_biomarkers, d_z=1, n_subtypes=num_sets).to(DEVICE)

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
    subtype_accuracy = compute_subtype_accuracy_with_cluster_mapping(vae, dataset_names)
    print("Subtype accuracy:", subtype_accuracy)

    # Visualise stage predictions against ground truth
    fig, ax = predicted_stage_comparison(train_loader, num_biomarkers, vae, DEVICE)
    fig.show()

    # Visualise the training set encoded in the latent space
    fig, ax = plot_dataset_in_latent_space(net=vae, dataset_names=dataset_names)

    fig.show()
    plt.show()


    for X, y in val_loader:
        print(torch.column_stack([vae.enc(X)[2], y]))  # q(y|x), true stage label
