# --------------------------- 2 LATENT VARIABLES EXPERIMENT ---------------------------
import os
import torch
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import evaluation
from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_OBS_VAL_DIR, SIMULATED_LABEL_VAL_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec

from models import ae_stager, vae_stager


def run_training(n_epochs, net, model_name, train_loader, optimiser, criterion, model_type, device):
    dataset_size = len(train_loader.dataset)
    model_dir = os.path.join(SAVED_MODEL_DIR, model_type)
    os.makedirs(model_dir, exist_ok=True)

    enc_path = os.path.join(model_dir, "enc_" + model_name + ".pth")
    dec_path = os.path.join(model_dir, "dec_" + model_name + ".pth")

    epochs_without_improvement = 0
    best_reconstruction_error = float('inf')
    epoch_patience = 20
    minimum_improvement = 1e-4  # Stop training if improvement falls below this

    for epoch in tqdm(range(n_epochs), desc=f"Training {model_type}"):
        train_loss = 0.0
        for (X, _) in train_loader:
            X = X.to(device)
            optimiser.zero_grad()

            loss = criterion(X, net, device)

            # if torch.isnan(torch.tensor(loss)):
            #     print(loss)
            # if torch.any(torch.isnan(X)):
            #     print(X)

            loss.backward()

            # for param in net.enc.parameters():
            #     if torch.any(torch.isnan(param.grad)):
            #         print("gradient:")
            #         exit()

            optimiser.step()
            train_loss += loss.item() * X.shape[0] / dataset_size

            # for param in net.enc.parameters():
            #     if torch.any(torch.isnan(param)):
            #         print("parameter:")
            #         exit()

        # compute error between latents and stages - just to track progress
        # mse_stage_error, reconstruction_error = evaluate_autoencoder(train_loader, net, device)
        reconstruction_error = evaluation.compute_reconstruction_mse(train_loader, net, device)

        if reconstruction_error < best_reconstruction_error:
            # Compute the correct latent direction, but only if the current model is better than the preceding
            net.calculate_latent_direction(train_loader)

            reconstruction_improvement = best_reconstruction_error - reconstruction_error
            best_reconstruction_error = reconstruction_error
            torch.save(net.enc.state_dict(), enc_path)
            torch.save(net.dec.state_dict(), dec_path)
            epochs_without_improvement = 0

            # Terminate training early if reconstruction improvement is below minimum accepted improvement
            if reconstruction_improvement < minimum_improvement:
                print(f"Ending training early as observed improvement is now under {minimum_improvement}")
                return
        else:
            # Terminate training early if still no decrease in reconstruction error (evaluated every 10 epochs)
            epochs_without_improvement += 1

            if epochs_without_improvement >= epoch_patience:
                print(
                    f"Ending training early as no decrease in reconstruction error observed in {epoch_patience} epochs")
                return

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, train loss = {train_loss:.4f}, average reconstruction squared distance = {reconstruction_error:.4f}")


if __name__ == "__main__":
    # USER CONFIGURATION --------------------
    num_sets = 3
    dataset_names = [f"synthetic_120_10_{i}" for i in range(num_sets)]
    model_name = "synthetic_120_10_0"

    model_type = "ae"  # only vae or ae supported currently
    if model_type not in ["vae", "ae"]:
        print("Model type must be one of 'vae' or 'ae' (case-sensitive)")
        exit()
    # ---------------------------------------
    # Load training and validation sets
    train_dataset = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_VAL_DIR, label_directory=SIMULATED_LABEL_VAL_DIR)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

    example_x, _ = next(iter(train_loader))
    num_biomarkers = example_x.shape[1]

    # Instantiate saved_models
    enc = None
    dec = None

    if model_type == "vae":
        enc = vae_stager.Encoder(d_in=num_biomarkers, d_latent=2).to(DEVICE)
        dec = vae_stager.Decoder(d_out=num_biomarkers, d_latent=2).to(DEVICE)

        net = vae_stager.VAE(enc=enc, dec=dec)
        criterion = vae_stager.vae_criterion

    elif model_type == "ae":
        enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=2).to(DEVICE)
        dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=2).to(DEVICE)

        net = ae_stager.AE(enc=enc, dec=dec)
        criterion = ae_stager.ae_criterion

    opt = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=0.001)

    if enc is None or dec is None:
        print("Arguments not validated properly - please fix")
        exit()

    train_set = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)

    val_set = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_VAL_DIR, label_directory=SIMULATED_LABEL_VAL_DIR)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True)

    num_biomarkers = next(iter(train_loader))[0].shape[1]

    n_epochs = 1000

    # ---------- Train -----------
    # Run training loop
    run_training(n_epochs, net, model_name, train_loader, opt, criterion, model_type=model_type, device=DEVICE)

    # Evaluate on the final models saved during training
    enc_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "enc_" + model_name + ".pth")
    dec_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "dec_" + model_name + ".pth")

    enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
    dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

    reconstruction_mse = evaluation.compute_reconstruction_mse(val_loader, net, DEVICE)

    print("Reconstruction MSE of trained model on validation set:", reconstruction_mse)

    # exit()

    # Visualise the latent space -------------------------
    n_rows = 5
    n_cols = 5
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 10))
    fig.suptitle("Visualisation of decoded latent space")

    latent_grid = torch.linspace(0., 1., 5, requires_grad=False).to(DEVICE)
    latent_grid = torch.stack(torch.meshgrid(latent_grid, latent_grid), dim=-1)

    for i in range(n_rows * n_cols):
        y = i // n_rows
        x = i % n_cols
        latents = latent_grid[y][x]
        outputs = net.decode(latents).detach().cpu()
        ax.flat[i].imshow(outputs.unsqueeze(0), cmap='grey')
        ax.flat[i].set_title(f"{latents.tolist()}")

        ax.flat[i].set_yticks([])
        ax.flat[i].set_xticks(range(num_biomarkers))

    fig.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("Encoding of the training set in the latent space")

    latents = []
    labels = []
    for X, stage in train_loader:
        latents.append(net.encode(X))
        labels.append(stage)
    latents = torch.concatenate(latents, dim=0).detach().cpu()
    labels = torch.concatenate(labels, dim=0).detach().cpu()

    pcm = ax.scatter(latents[:, 0], latents[:, 1], c=labels)
    fig.colorbar(pcm, ax=ax)
    fig.show()

    plt.show()


