# A script to generate datasets of different biomarker counts, measure the time taken to learn stages from them
# as well as run inference (compute latent representations) on the whole training set),
# and plot time taken against biomarker count.

import torch
import itertools
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from datasets.simulate_data import generate_data
from train_autoencoder import run_training
from datasets.synthetic_dataset import SyntheticDataset
from models import ae_stager, vae_stager
from evaluation import evaluate_autoencoder
from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_TEST_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_TEST_DIR


if __name__ == "__main__":
    # Generate sets of data with varying sizes
    # n_biomarkers_all = [i for i in range(50, 505, 50)]
    # n_biomarkers_all = [i for i in range(10, 40, 10)]
    n_biomarkers_all = [10, 50, 100, 250, 500]

    # Define dataset names
    dataset_names = [f"synthetic_{12 * n_biomarkers}_{n_biomarkers}_0" for n_biomarkers in n_biomarkers_all]

    # For each dataset, fit a model and measure how long it takes, then evaluate reconstruction error.
    time_taken_vae = []
    time_taken_ae = []
    reconstruction_errors_vae = []
    reconstruction_errors_ae = []
    staging_errors_vae = []
    staging_errors_ae = []

    # Number of trials to run on each dataset size, to later be averaged over
    n_trials = 1

    # Write out results to file in case training is interrupted
    with open('../dataset_size_comparison_results.csv', 'w', newline='') as output_f:
        writer = csv.writer(output_f)
        writer.writerow(['mean_vae_time', 'vae_time_std', 'mean_ae_time', 'ae_time_std',
                         'mean_vae_reconstruction', 'vae_reconstruction_std',
                         'mean_ae_reconstruction', 'ae_reconstruction_std',
                         'mean_vae_staging', 'vae_staging_std',
                         'mean_ae_staging', 'ae_staging_std'])

        for i in range(len(dataset_names)):
            time_taken_vae.append(np.zeros(n_trials))
            time_taken_ae.append(np.zeros(n_trials))
            reconstruction_errors_vae.append(np.zeros(n_trials))
            reconstruction_errors_ae.append(np.zeros(n_trials))
            staging_errors_vae.append(np.zeros(n_trials))
            staging_errors_ae.append(np.zeros(n_trials))

            for trial in range(n_trials):
                n_biomarkers = n_biomarkers_all[i]
                dataset_name = dataset_names[i]

                # Generate a fresh dataset
                generate_data(n_biomarkers=n_biomarkers, n_mci=10 * n_biomarkers,
                              n_controls=n_biomarkers, n_patients=n_biomarkers)

                train_set = SyntheticDataset(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TRAIN_DIR,
                                             label_directory=SIMULATED_LABEL_TRAIN_DIR)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
                val_set = SyntheticDataset(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TEST_DIR,
                                           label_directory=SIMULATED_LABEL_TEST_DIR)
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True)

                n_epochs = 1000

                print(f"Now training on dataset with {n_biomarkers} biomarkers ({trial + 1}/{n_trials} trials)")

                # ---------- Train VAE -----------
                # Define network
                vae_enc = vae_stager.Encoder(d_in=n_biomarkers, d_latent=1).to(DEVICE)
                vae_dec = vae_stager.Decoder(d_out=n_biomarkers, d_latent=1).to(DEVICE)

                vae = vae_stager.VAE(enc=vae_enc, dec=vae_dec)

                # Define optimiser
                opt_vae = torch.optim.Adam(itertools.chain(vae_enc.parameters(), vae_dec.parameters()), lr=0.001)

                # Run training loop
                start_time = time.time()
                run_training(n_epochs, vae, dataset_name, train_loader, val_loader,
                             opt_vae, vae_stager.vae_criterion_wrapper(beta=1), model_type="vae",
                             device=DEVICE)
                time_taken = time.time() - start_time  # in seconds
                time_taken_vae[-1][trial] = time_taken

                # Load best model obtained from training, to run inference and evaluate.
                vae_enc_model_path = os.path.join(SAVED_MODEL_DIR, "vae", "enc_" + dataset_name + ".pth")
                vae_dec_model_path = os.path.join(SAVED_MODEL_DIR, "vae", "dec_" + dataset_name + ".pth")

                vae_enc.load_state_dict(torch.load(vae_enc_model_path, map_location=DEVICE))
                vae_dec.load_state_dict(torch.load(vae_dec_model_path, map_location=DEVICE))

                # Measure the time taken to run inference and compute performance metrics on the validation set.
                start_time = time.time()
                staging_mse, reconstruction_mse = evaluate_autoencoder(val_loader, vae, DEVICE)
                time_taken = time.time() - start_time  # in seconds
                time_taken_vae[-1][trial] += time_taken

                reconstruction_errors_vae[-1][trial] = reconstruction_mse.cpu()
                staging_errors_vae[-1][trial] = staging_mse.cpu()

                # ---------- Train AE -----------
                ae_enc = ae_stager.Encoder(d_in=n_biomarkers, d_latent=1).to(DEVICE)
                ae_dec = ae_stager.Decoder(d_out=n_biomarkers, d_latent=1).to(DEVICE)

                ae = ae_stager.AE(enc=ae_enc, dec=ae_dec)

                opt_ae = torch.optim.Adam(itertools.chain(ae_enc.parameters(), ae_dec.parameters()), lr=0.001)

                # Run training loop
                start_time = time.time()
                run_training(n_epochs, ae, dataset_name, train_loader, val_loader,
                             opt_ae, ae_stager.ae_criterion, model_type="ae",
                             device=DEVICE)
                time_taken = time.time() - start_time  # in seconds
                time_taken_ae[-1][trial] = time_taken

                # Load best model obtained from training, to run inference and evaluate.
                ae_enc_model_path = os.path.join(SAVED_MODEL_DIR, "ae", "enc_" + dataset_name + ".pth")
                ae_dec_model_path = os.path.join(SAVED_MODEL_DIR, "ae", "dec_" + dataset_name + ".pth")

                ae_enc.load_state_dict(torch.load(ae_enc_model_path, map_location=DEVICE))
                ae_dec.load_state_dict(torch.load(ae_dec_model_path, map_location=DEVICE))

                # Measure the time taken to run inference and compute performance metrics on the validation set.
                start_time = time.time()
                staging_mse, reconstruction_mse = evaluate_autoencoder(val_loader, ae, DEVICE)
                time_taken = time.time() - start_time  # in seconds
                time_taken_ae[-1][trial] += time_taken

                reconstruction_errors_ae[-1][trial] = reconstruction_mse.cpu()
                staging_errors_ae[-1][trial] = staging_mse.cpu()

            # ----------- Write results to file -----------
            writer.writerow([time_taken_vae[-1].mean().item(),
                             time_taken_vae[-1].std().item(),
                             time_taken_ae[-1].mean().item(),
                             time_taken_ae[-1].std().item(),
                             reconstruction_errors_vae[-1].mean().item(),
                             reconstruction_errors_vae[-1].std().item(),
                             reconstruction_errors_ae[-1].mean().item(),
                             reconstruction_errors_ae[-1].std().item(),
                             staging_errors_vae[-1].mean().item(),
                             staging_errors_vae[-1].std().item(),
                             staging_errors_ae[-1].mean().item(),
                             staging_errors_ae[-1].std().item()])
            output_f.flush()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(time_taken_vae).mean(axis=-1))
    ax.errorbar(np.array(n_biomarkers_all), np.array(time_taken_vae).mean(axis=-1),
                yerr=np.array(time_taken_vae).std(axis=-1),
                fmt='o',
                capsize=3)
    fig.suptitle("Full training + inference time of the VAE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Time taken (s)")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(time_taken_ae).mean(axis=-1))
    ax.errorbar(np.array(n_biomarkers_all), np.array(time_taken_ae).mean(axis=-1),
                yerr=np.array(time_taken_ae).std(axis=-1),
                fmt='o',
                capsize=3)
    fig.suptitle("Full training + inference time of the AE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Time taken (s)")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(reconstruction_errors_vae).mean(axis=-1))
    ax.errorbar(np.array(n_biomarkers_all), np.array(reconstruction_errors_vae).mean(axis=-1),
                yerr=np.array(reconstruction_errors_vae).std(axis=-1),
                fmt='o',
                capsize=3)
    fig.suptitle("Mean squared reconstruction error of VAE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Reconstruction error")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(reconstruction_errors_ae).mean(axis=-1))
    ax.errorbar(np.array(n_biomarkers_all), np.array(reconstruction_errors_ae).mean(axis=-1),
                yerr=np.array(reconstruction_errors_ae).std(axis=-1),
                fmt='o',
                capsize=3)
    fig.suptitle("Mean squared reconstruction error of AE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Reconstruction error")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(staging_errors_vae).mean(axis=-1))
    ax.errorbar(np.array(n_biomarkers_all), np.array(staging_errors_vae).mean(axis=-1),
                yerr=np.array(staging_errors_vae).std(axis=-1),
                fmt='o',
                capsize=3)
    fig.suptitle("Root mean squared staging error of VAE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Staging error")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(staging_errors_ae).mean(axis=-1))
    ax.errorbar(np.array(n_biomarkers_all), np.array(staging_errors_ae).mean(axis=-1),
                yerr=np.array(staging_errors_ae).std(axis=-1),
                fmt='o',
                capsize=3)
    fig.suptitle("Root mean squared staging error of AE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Staging error")
    fig.show()

    plt.show()
