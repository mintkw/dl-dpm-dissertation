import os

import torch
import torch.distributions as dist
import torch.nn as nn
import itertools
from tqdm import tqdm

from config import DEVICE, MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_VAL_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_VAL_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from evaluation import evaluate_autoencoder

import vae_stager
import ae_stager


def run_training(n_epochs, net, dataset_size, train_loader, optimiser, criterion, model_name, device):
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    enc_path = os.path.join(model_dir, "enc_" + dataset_name + ".pth")
    dec_path = os.path.join(model_dir, "dec_" + dataset_name + ".pth")

    epochs_without_improvement = 0
    best_loss = float('inf')
    epoch_patience = 10
    minimum_improvement = 1e-4  # Stop training if improvement falls below this

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

        # compute error between latents and stages - just to track progress
        mse_stage_error, reconstruction_error = evaluate_autoencoder(train_loader, net, device)

        if train_loss < best_loss:
            # Save the model with the current lowest loss
            torch.save(net.enc.state_dict(), enc_path)
            torch.save(net.dec.state_dict(), dec_path)
            epochs_without_improvement = 0

            # Terminate training early if reconstruction improvement is below minimum accepted improvement
            if best_loss - train_loss < minimum_improvement:
                print(f"Ending training early as observed loss decrease is now under {minimum_improvement}")
                return

            best_loss = train_loss

        else:
            # Terminate training early if still no decrease in reconstruction error (evaluated every 10 epochs)
            epochs_without_improvement += 1

            if epochs_without_improvement >= epoch_patience:
                print(
                    f"Ending training early as no decrease in loss observed in {epoch_patience} epochs")
                return

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, train loss = {train_loss:.4f}, average reconstruction squared distance = {reconstruction_error:.4f}, MSE stage error = {mse_stage_error:.4f}")


if __name__ == "__main__":
    dataset_name = "synthetic_120_10_dpm_same"

    train_set = SyntheticDatasetVec(dataset_name=dataset_name, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)

    # val_set = SyntheticDatasetVec(dataset_name=dataset_name, obs_directory=SIMULATED_OBS_VAL_DIR, label_directory=SIMULATED_LABEL_VAL_DIR)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True)

    example_x, _ = next(iter(train_loader))
    num_biomarkers = example_x.shape[1]

    n_epochs = 1000

    # ---------- Train VAE -----------
    # Define network
    vae_enc = vae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    vae_dec = vae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

    vae = vae_stager.VAE(enc=vae_enc, dec=vae_dec)

    # Define optimiser
    opt_vae = torch.optim.Adam(itertools.chain(vae_enc.parameters(), vae_dec.parameters()), lr=0.0005)

    # Run training loop
    def vae_criterion(X, vae, device):
        elbos = vae_stager.compute_elbo(vae, X, device)

        # The loss is the sum of the negative per-datapoint ELBO
        loss = -elbos.sum()

        return loss

    run_training(n_epochs, vae, len(train_set), train_loader, opt_vae, vae_criterion, model_name="vae", device=DEVICE)

    # Evaluate
    staging_mse, reconstruction_mse = evaluate_autoencoder(train_loader, vae, DEVICE)
    print("Staging MSE of trained VAE:", staging_mse)
    print("Reconstruction MSE of trained VAE:", reconstruction_mse)

    # ---------- Train AE -----------
    ae_enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    ae_dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

    ae = ae_stager.AE(enc=ae_enc, dec=ae_dec)

    opt_ae = torch.optim.Adam(itertools.chain(ae_enc.parameters(), ae_dec.parameters()), lr=0.001)

    # Run training loop
    def ae_criterion(X, ae, device):
        reconstructions = ae.reconstruct(X)
        latents = ae.encode(X)

        rms_error = torch.sqrt((X - reconstructions) ** 2).sum()

        # use mean correlation as a way to regularise the direction of the latent (lower for lower biomarker values)
        correlation_matrix = torch.corrcoef(torch.transpose(torch.concat([X, latents], dim=1), 1, 0))
        mean_correlation = correlation_matrix[-1][:-1].mean()

        return rms_error - mean_correlation

    run_training(n_epochs, ae, len(train_set), train_loader, opt_ae, ae_criterion, model_name="ae", device=DEVICE)

    # Evaluate
    staging_mse, reconstruction_mse = evaluate_autoencoder(train_loader, ae, DEVICE)
    print("Staging MSE of trained AE:", staging_mse)
    print("Reconstruction MSE of trained AE:", reconstruction_mse)
