# Experiments with using a multi-dimensional latent space to capture different subtypes
# present in input data.

import os
import torch
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

import evaluation
from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_OBS_TEST_DIR, SIMULATED_LABEL_TEST_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from train_autoencoder import run_training

from models import ae_stager, vae_stager


def plot_dataset_in_latent_space(net, dataset_names):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("Encoding of the training set in the latent space")

    # Load the data from each subtype individually so they can be differentiated through colour coding
    train_datasets = [SyntheticDatasetVec(dataset_names=[dataset_name],
                                          obs_directory=SIMULATED_OBS_TRAIN_DIR,
                                          label_directory=SIMULATED_LABEL_TRAIN_DIR) for dataset_name in dataset_names]
    train_loaders = [torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True) for dataset in train_datasets]
    cmaps = ['viridis', 'inferno']

    for i in range(len(train_loaders)):
        loader = train_loaders[i]
        stage_latents = []
        subtype_latents = []
        labels = []
        for X, stage in loader:
            stage_latents.append(net.predict_stage(X))
            subtype_latents.append(net.predict_subtype(X))
            # latents.append(net.encode(X))
            labels.append(stage)
        # latents = torch.concatenate(latents, dim=0).detach().cpu()
        stage_latents = torch.concatenate(stage_latents, dim=0).detach().cpu()
        subtype_latents = torch.concatenate(subtype_latents, dim=0).detach().cpu()

        labels = torch.concatenate(labels, dim=0).detach().cpu()

        colourmap = ax.scatter(stage_latents, subtype_latents, c=labels, cmap=cmaps[i])
        fig.colorbar(colourmap, ax=ax)

    fig.show()


class AE(ae_stager.AE):
    def __init__(self, enc, dec):
        super().__init__(enc, dec)

    def predict_uncorrected_stage(self, X):
        return self.enc(X)[:, 0].unsqueeze(1)

    def predict_subtype(self, X):
        return self.enc(X)[:, 1]


class VAE(vae_stager.VAE):
    def __init__(self, enc, dec):
        super().__init__(enc, dec)

    def predict_uncorrected_stage(self, X):
        return self.enc(X).mean[:, 0].unsqueeze(1)

    def predict_subtype(self, X):
        return self.enc(X).mean[:, 1]


if __name__ == "__main__":
    # USER CONFIGURATION --------------------
    num_sets = 2
    dataset_names = [f"synthetic_120_10_{i}" for i in range(num_sets)]
    model_name = "synthetic_120_10_0"

    model_type = "vae"  # only vae or ae supported currently
    if model_type not in ["vae", "ae"]:
        print("Model type must be one of 'vae' or 'ae' (case-sensitive)")
        exit()
    # ---------------------------------------
    # Load training and validation sets
    train_dataset = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TEST_DIR, label_directory=SIMULATED_LABEL_TEST_DIR)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

    example_x, _ = next(iter(train_loader))
    num_biomarkers = example_x.shape[1]

    # Instantiate saved_models
    enc = None
    dec = None

    if model_type == "vae":
        enc = vae_stager.Encoder(d_in=num_biomarkers, d_latent=2).to(DEVICE)
        dec = vae_stager.Decoder(d_out=num_biomarkers, d_latent=2).to(DEVICE)

        net = VAE(enc=enc, dec=dec)
        criterion = vae_stager.vae_criterion_wrapper(beta=2)

    elif model_type == "ae":
        enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=2).to(DEVICE)
        dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=2).to(DEVICE)

        net = AE(enc=enc, dec=dec)
        criterion = ae_stager.ae_criterion

    opt = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=0.001)

    if enc is None or dec is None:
        print("Arguments not validated properly - please fix")
        exit()

    train_set = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)

    val_set = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TEST_DIR, label_directory=SIMULATED_LABEL_TEST_DIR)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True)

    num_biomarkers = next(iter(train_loader))[0].shape[1]

    n_epochs = 1000

    # ---------- Train -----------
    # Run training loop
    run_training(n_epochs, net, model_name, train_loader, val_loader, opt, criterion, model_type=model_type, device=DEVICE)

    # Evaluate on the final models saved during training
    enc_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "enc_" + model_name + ".pth")
    dec_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "dec_" + model_name + ".pth")

    enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
    dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

    reconstruction_mse = evaluation.compute_reconstruction_mse(val_loader, net, DEVICE)

    print("Reconstruction MSE of trained model on validation set:", reconstruction_mse)

    # Visualise the decoded latent space -------------------------
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
        outputs = net.decode_latent(latents).detach().cpu()
        ax.flat[i].imshow(outputs.unsqueeze(0), cmap='grey')
        ax.flat[i].set_title(f"{latents.tolist()}")

        ax.flat[i].set_yticks([])
        ax.flat[i].set_xticks(range(num_biomarkers))

    fig.show()

    # Visualise the training set encoded in the latent space ---------------------
    plot_dataset_in_latent_space(net=net, dataset_names=dataset_names)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # fig.suptitle("Encoding of the training set in the latent space")
    #
    # # Load the data from each subtype individually so they can be differentiated through colour coding
    # train_datasets = [SyntheticDatasetVec(dataset_names=[dataset_name],
    #                                       obs_directory=SIMULATED_OBS_TRAIN_DIR,
    #                                       label_directory=SIMULATED_LABEL_TRAIN_DIR) for dataset_name in dataset_names]
    # train_loaders = [torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True) for dataset in train_datasets]
    # cmaps = ['viridis', 'inferno']
    #
    # for i in range(len(train_loaders)):
    #     loader = train_loaders[i]
    #     latents = []
    #     labels = []
    #     for X, stage in loader:
    #         latents.append(net.predict_stage(X))
    #         labels.append(stage)
    #     latents = torch.concatenate(latents, dim=0).detach().cpu()
    #     labels = torch.concatenate(labels, dim=0).detach().cpu()
    #
    #     colourmap = ax.scatter(latents[:, 0], latents[:, 1], c=labels, cmap=cmaps[i])
    #     fig.colorbar(colourmap, ax=ax)
    #
    # fig.show()

    plt.show()


