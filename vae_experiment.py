import os

import torch
import torch.distributions as dist
import torch.nn as nn
import itertools
from tqdm import tqdm

from config import SIMULATED_OBSERVATIONS_DIR, SIMULATED_LABELS_DIR, DEVICE, VAE_MODEL_DIR
from datasets.synthetic_dataset_matrix import SyntheticDatasetMat
from datasets.synthetic_dataset_vector import SyntheticDatasetVec


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
        mu = self.fc_mu(h)
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


def compute_elbo(enc, dec, X, device):
    # q(z | X)
    q_z = enc(X)

    # sample a vector z from q(z | X), using the reparameterisation trick
    epsilon = dist.Normal(0, 1).sample(q_z.mean.shape).to(device)  # not sure if this is right lol
    z = q_z.mean + epsilon * torch.sqrt(q_z.variance)

    # log prior : log p(z_i), chosen to be a standard multivariate gaussian
    log_prior = dist.Normal(0, 1).log_prob(z).sum(-1)

    # log posterior : log p(x_i | z_i)
    log_posterior = dec(z).log_prob(X).sum(-1).sum(-1).sum(-1)

    # log q(z_i | x_i)
    log_qz = q_z.log_prob(z).sum(-1)

    return log_prior + log_posterior - log_qz


def train_vae(N_epochs, enc, dec, train_loader, train_dataset, optimiser, dataset_name, device):
    lowest_reconstruction_error = float('inf')
    enc_path = os.path.join(VAE_MODEL_DIR, "enc_" + dataset_name + ".pth")
    dec_path = os.path.join(VAE_MODEL_DIR, "dec_" + dataset_name + ".pth")

    # Create necessary directories
    os.makedirs(VAE_MODEL_DIR, exist_ok=True)

    for epoch in tqdm(range(N_epochs), desc="Training VAE"):
        train_loss = 0.0
        for (X, _) in train_loader:
            X = X.to(device)
            opt_vae.zero_grad()

            elbos = compute_elbo(enc, dec, X, device)

            # The loss is the sum of the negative per-datapoint ELBO
            loss = -elbos.sum()
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * X.shape[0] / len(train_dataset)

        if epoch % 10 == 0:
            # compute error between latents and stages - just to track progress
            mse_stage_error, reconstruction_error = evaluate(train_dataloader, enc, dec, device)

            if reconstruction_error < lowest_reconstruction_error:
                lowest_reconstruction_error = min(reconstruction_error.item(), lowest_reconstruction_error)
                torch.save(enc.state_dict(), enc_path)
                torch.save(dec.state_dict(), dec_path)

            print(f"Epoch {epoch}, train loss = {train_loss:.4f}, average reconstruction squared distance = {reconstruction_error:.4f}, MSE stage error = {mse_stage_error:.4f}")


def evaluate(dataloader, enc, dec, device):
    predictions = []
    gt_stages = []
    reconstruction_errors = []
    with torch.no_grad():
        for X, label in dataloader:
            gt_stages.append(label)

            X = X.to(device)
            raw_preds = enc(X).mean  # take mean as preds or sample?
            predictions.append(raw_preds)

            reconstruction_errors.append((dec(enc(X).sample()).sample() - X) ** 2)

    # scale predictions
    predictions = torch.concatenate(predictions).squeeze().to(device)
    predictions = (predictions - torch.min(predictions)) / (torch.max(predictions) - torch.min(predictions))
    # predictions = torch.sigmoid(predictions)

    # scale stages
    gt_stages = torch.concatenate(gt_stages).to(device)
    gt_stages /= torch.max(gt_stages)

    reconstruction_errors = torch.concatenate(reconstruction_errors).to(device)

    # latent can scale in an opposite direction to stages, so try both directions and take the min error
    mse_stage_error = torch.min(torch.mean((predictions - gt_stages) ** 2),
                                torch.mean((predictions - 1 + gt_stages) ** 2))

    reconstruction_error = torch.mean(reconstruction_errors)

    return mse_stage_error, reconstruction_error


if __name__ == "__main__":
    dataset_name = "synthetic_1200_50_dpm_0"
    # VAE trying to infer stage with latent space
    train_set = SyntheticDatasetVec(dataset_name=dataset_name)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)

    example_x, _ = next(iter(train_dataloader))
    num_biomarkers = example_x.shape[1]

    enc = Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    dec = Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

    opt_vae = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=0.001)

    # Run training loop
    n_epochs = 200
    train_vae(n_epochs, enc, dec, train_dataloader, train_set, opt_vae, dataset_name, DEVICE)

    # Evaluate by computing MSE error
    print("Mean squared error:", evaluate(train_dataloader, enc, dec, DEVICE)[0])

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
