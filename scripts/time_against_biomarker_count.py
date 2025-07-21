# A script to generate datasets of different biomarker counts, measure the time taken to learn stages from them,
# and plot time taken against biomarker count.

import torch
import itertools
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from datasets.simulate_data import generate_normalised_data
from train_autoencoder import run_training
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from models import ae_stager, vae_stager
from evaluation import evaluate_autoencoder
from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_VAL_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_VAL_DIR


if __name__ == "__main__":
    # Generate sets of data with varying sizes
    # n_biomarkers_all = [i for i in range(50, 505, 50)]
    n_biomarkers_all = [50]
    # n_biomarkers_all = [i for i in range(5, 15, 5)]
    # for n_biomarkers in n_biomarkers_all:
        # generate_normalised_data(n_biomarkers=n_biomarkers, n_mci=10 * n_biomarkers,
        #                          n_controls=n_biomarkers, n_patients=n_biomarkers)

    # Define dataset names
    dataset_names = [f"synthetic_{12 * n_biomarkers}_{n_biomarkers}_0" for n_biomarkers in n_biomarkers_all]

    # For each dataset, fit a model and measure how long it takes, then evaluate reconstruction error.
    times_taken_to_fit_vae = []
    times_taken_to_fit_ae = []
    reconstruction_errors_vae = []
    reconstruction_errors_ae = []
    staging_errors_vae = []
    staging_errors_ae = []

    # Number of trials to run on each dataset size, to later be averaged over
    n_trials = 5

    # Write out results to file in case training is interrupted
    with open('../dataset_size_comparison_results.csv', 'w', newline='') as output_f:
        writer = csv.writer(output_f)
        writer.writerow(['vae_time', 'ae_time', 'vae_reconstruction', 'ae_reconstruction', 'vae_staging', 'ae_staging'])

        for i in range(len(dataset_names)):
            times_taken_to_fit_vae.append(0.)
            times_taken_to_fit_ae.append(0.)
            reconstruction_errors_vae.append(0.)
            reconstruction_errors_ae.append(0.)
            staging_errors_vae.append(0.)
            staging_errors_ae.append(0.)

            for trial in range(n_trials):
                n_biomarkers = n_biomarkers_all[i]
                dataset_name = dataset_names[i]

                # Generate a fresh dataset
                generate_normalised_data(n_biomarkers=n_biomarkers, n_mci=10 * n_biomarkers,
                                         n_controls=n_biomarkers, n_patients=n_biomarkers)

                train_set = SyntheticDatasetVec(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TRAIN_DIR,
                                                label_directory=SIMULATED_LABEL_TRAIN_DIR)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
                val_set = SyntheticDatasetVec(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_VAL_DIR,
                                              label_directory=SIMULATED_LABEL_VAL_DIR)
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
                run_training(n_epochs, vae, dataset_name, train_loader, opt_vae, vae_stager.vae_criterion, model_type="vae",
                             device=DEVICE)
                time_taken = time.time() - start_time  # in seconds
                times_taken_to_fit_vae[-1] += time_taken / n_trials

                # Evaluate
                vae_enc_model_path = os.path.join(SAVED_MODEL_DIR, "vae", "enc_" + dataset_name + ".pth")
                vae_dec_model_path = os.path.join(SAVED_MODEL_DIR, "vae", "dec_" + dataset_name + ".pth")

                vae_enc.load_state_dict(torch.load(vae_enc_model_path, map_location=DEVICE))
                vae_dec.load_state_dict(torch.load(vae_dec_model_path, map_location=DEVICE))

                staging_mse, reconstruction_mse = evaluate_autoencoder(val_loader, vae, DEVICE)
                reconstruction_mse = reconstruction_mse.cpu()
                reconstruction_errors_vae[-1] += reconstruction_mse.cpu() / n_trials
                staging_errors_vae[-1] += staging_mse.cpu() / n_trials

                # ---------- Train AE -----------
                ae_enc = ae_stager.Encoder(d_in=n_biomarkers, d_latent=1).to(DEVICE)
                ae_dec = ae_stager.Decoder(d_out=n_biomarkers, d_latent=1).to(DEVICE)

                ae = ae_stager.AE(enc=ae_enc, dec=ae_dec)

                opt_ae = torch.optim.Adam(itertools.chain(ae_enc.parameters(), ae_dec.parameters()), lr=0.001)

                # Run training loop
                start_time = time.time()
                run_training(n_epochs, ae, dataset_name, train_loader, opt_ae, ae_stager.ae_criterion, model_type="ae",
                             device=DEVICE)
                time_taken = time.time() - start_time  # in seconds
                times_taken_to_fit_ae[-1] += time_taken / n_trials

                # Evaluate
                ae_enc_model_path = os.path.join(SAVED_MODEL_DIR, "ae", "enc_" + dataset_name + ".pth")
                ae_dec_model_path = os.path.join(SAVED_MODEL_DIR, "ae", "dec_" + dataset_name + ".pth")

                ae_enc.load_state_dict(torch.load(ae_enc_model_path, map_location=DEVICE))
                ae_dec.load_state_dict(torch.load(ae_dec_model_path, map_location=DEVICE))

                staging_mse, reconstruction_mse = evaluate_autoencoder(val_loader, ae, DEVICE)
                reconstruction_errors_ae[-1] += reconstruction_mse.cpu() / n_trials
                staging_errors_ae[-1] += staging_mse.cpu() / n_trials

            # ----------- Write results to file -----------
            writer.writerow([times_taken_to_fit_vae[-1],
                             times_taken_to_fit_ae[-1],
                             reconstruction_errors_vae[-1].item(),
                             reconstruction_errors_ae[-1].item(),
                             staging_errors_vae[-1].item(),
                             staging_errors_ae[-1].item()])
            output_f.flush()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(times_taken_to_fit_vae))
    fig.suptitle("Time taken to fit VAE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Time taken (s)")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(times_taken_to_fit_ae))
    fig.suptitle("Time taken to fit AE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Time taken (s)")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(reconstruction_errors_vae))
    fig.suptitle("Mean squared reconstruction error of VAE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Reconstruction error")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(reconstruction_errors_ae))
    fig.suptitle("Mean squared reconstruction error of AE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Reconstruction error")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(staging_errors_vae))
    fig.suptitle("Mean squared staging error of VAE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Staging error")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(staging_errors_ae))
    fig.suptitle("Mean squared staging error of AE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Staging error")
    fig.show()

    plt.show()