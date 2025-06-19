import os

import torch
import torch.distributions as dist
import torch.nn as nn
import itertools
from tqdm import tqdm

from config import SIMULATED_OBSERVATIONS_DIR, SIMULATED_LABELS_DIR, DEVICE, MODEL_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from evaluation import evaluate_autoencoder

import vae_stager
import ae_stager


def run_training(n_epochs, net, dataset_size, train_loader, optimiser, criterion, model_name, device):
    lowest_reconstruction_error = float('inf')
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    enc_path = os.path.join(model_dir, "enc_" + dataset_name + ".pth")
    dec_path = os.path.join(model_dir, "dec_" + dataset_name + ".pth")

    for epoch in tqdm(range(n_epochs), desc=f"Training {model_name}"):
        train_loss = 0.0
        for (X, _) in train_loader:
            X = X.to(device)
            optimiser.zero_grad()

            # The loss is the sum of the negative per-datapoint ELBO
            loss = criterion(X, net, device)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * X.shape[0] / dataset_size

        if epoch % 10 == 0:
            # compute error between latents and stages - just to track progress
            mse_stage_error, reconstruction_error = evaluate_autoencoder(train_loader, net, device)

            if reconstruction_error < lowest_reconstruction_error:
                lowest_reconstruction_error = min(reconstruction_error.item(), lowest_reconstruction_error)
                torch.save(net.enc.state_dict(), enc_path)
                torch.save(net.dec.state_dict(), dec_path)

            print(f"Epoch {epoch}, train loss = {train_loss:.4f}, average reconstruction squared distance = {reconstruction_error:.4f}, MSE stage error = {mse_stage_error:.4f}")



if __name__ == "__main__":
    # dataset_name = "synthetic_600_30_dpm_0"
    dataset_name = "synthetic_120_10_dpm_0"

    train_set = SyntheticDatasetVec(dataset_name=dataset_name)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)

    example_x, _ = next(iter(train_loader))
    num_biomarkers = example_x.shape[1]

    n_epochs = 50

    # ---------- Train VAE -----------
    # Define network
    vae_enc = vae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    vae_dec = vae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

    vae = vae_stager.VAE(enc=vae_enc, dec=vae_dec)

    # Define optimiser
    opt_vae = torch.optim.Adam(itertools.chain(vae_enc.parameters(), vae_dec.parameters()), lr=0.001)

    # Run training loop
    # todo: can i make the following a lambda function
    def vae_criterion(X, vae, device):
        elbos = vae_stager.compute_elbo(vae, X, device)

        # The loss is the sum of the negative per-datapoint ELBO
        loss = -elbos.sum()

        return loss

    run_training(n_epochs, vae, len(train_set), train_loader, opt_vae, vae_criterion, model_name="vae", device=DEVICE)

    # Evaluate by computing MSE error
    print("Mean squared error of trained VAE:", evaluate_autoencoder(train_loader, vae, DEVICE)[0])

    # ---------- Train AE -----------
    ae_enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    ae_dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

    ae = ae_stager.AE(enc=ae_enc, dec=ae_dec)

    opt_ae = torch.optim.Adam(itertools.chain(ae_enc.parameters(), ae_dec.parameters()), lr=0.001)

    # Run training loop
    # todo: can i make the following a lambda function
    def ae_criterion(X, ae, device):
        reconstructions = ae.reconstruct(X)
        loss = ((X - reconstructions) ** 2).sum()

        return loss
    run_training(n_epochs, ae, len(train_set), train_loader, opt_ae, ae_criterion, model_name="ae", device=DEVICE)

    # Evaluate by computing MSE error
    print("Mean squared error of trained AE:", evaluate_autoencoder(train_loader, ae, DEVICE)[0])
