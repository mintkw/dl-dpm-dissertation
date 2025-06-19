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

if __name__ == "__main__":
    dataset_name = "synthetic_120_10_dpm_0"

    train_set = SyntheticDatasetVec(dataset_name=dataset_name)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)

    example_x, _ = next(iter(train_dataloader))
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
    vae_stager.train_vae(n_epochs, vae, train_dataloader, train_set, opt_vae, dataset_name, DEVICE)

    # Evaluate by computing MSE error
    print("Mean squared error of trained VAE:", evaluate_autoencoder(train_dataloader, vae, DEVICE)[0])

    # ---------- Train AE -----------
    ae_enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    ae_dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

    ae = ae_stager.AE(enc=ae_enc, dec=ae_dec)

    opt_ae = torch.optim.Adam(itertools.chain(ae_enc.parameters(), ae_dec.parameters()), lr=0.001)

    # Run training loop
    ae_stager.train_ae(n_epochs, ae, train_dataloader, train_set, opt_ae, dataset_name, DEVICE)

    # Evaluate by computing MSE error
    print("Mean squared error of trained AE:", evaluate_autoencoder(train_dataloader, ae, DEVICE)[0])
