# A script to generate datasets of different biomarker counts, measure the time taken to learn stages from them
# as well as run inference (compute latent representations) on the whole training set),
# and plot time taken against biomarker count.
import json

import torch
import itertools
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split

from datasets.simulate_data import generate_data
from dpm_algorithms.train_autoencoder import run_training
from dpm_algorithms.evaluation import evaluate_autoencoder, evaluate_sequence
from dpm_algorithms.autoencoder_sequence_inference import infer_seq_from_network
from datasets.synthetic_dataset import SyntheticDataset
from models import ae_stager, vae_stager
from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_TEST_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_TEST_DIR


if __name__ == "__main__":
    # Generate sets of data with varying sizes
    n_biomarkers_all = [10, 50, 100, 250, 500]

    # For each dataset, fit a model and measure how long it takes, then evaluate.
    time_taken_vae = []
    time_taken_ae = []
    staging_errors_vae = []
    staging_errors_ae = []
    sequence_scores_vae = []
    sequence_scores_ae = []

    # Number of trials to run on each dataset size, to later be averaged over
    n_trials = 5

    # Write out results to file in case training is interrupted
    with open('scalability_with_dimensionality.csv', 'w', newline='') as output_f:
        writer = csv.writer(output_f)
        writer.writerow(['mean_vae_time', 'vae_time_std', 'mean_ae_time', 'ae_time_std',
                         'mean_vae_staging', 'vae_staging_std',
                         'mean_ae_staging', 'ae_staging_std',
                         'mean_vae_kt', 'vae_kt_std',
                         'mean_ae_kt', 'ae_kt_std'])

        for i in range(len(n_biomarkers_all)):
            n_biomarkers = n_biomarkers_all[i]

            time_taken_vae.append(np.zeros(n_trials))
            time_taken_ae.append(np.zeros(n_trials))
            staging_errors_vae.append(np.zeros(n_trials))
            staging_errors_ae.append(np.zeros(n_trials))
            sequence_scores_vae.append(np.zeros(n_trials))
            sequence_scores_ae.append(np.zeros(n_trials))

            # Generate datasets.
            generate_data(n_biomarkers=n_biomarkers, n_mci=10 * n_biomarkers,
                          n_controls=n_biomarkers, n_patients=n_biomarkers,
                          num_sets=n_trials)

            for trial in range(n_trials):
                dataset_name = f"synthetic_{12 * n_biomarkers}_{n_biomarkers}_{trial}"

                train_dataset = SyntheticDataset(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TRAIN_DIR,
                                                 label_directory=SIMULATED_LABEL_TRAIN_DIR)

                # Split training set
                train_indices, val_indices = train_test_split(range(len(train_dataset)), train_size=0.8)

                # Generate subset based on indices
                train_split = torch.utils.data.Subset(train_dataset, train_indices)
                val_split = torch.utils.data.Subset(train_dataset, val_indices)

                # Create dataloader
                train_loader = torch.utils.data.DataLoader(train_split, batch_size=16, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_split, batch_size=16, shuffle=True)

                test_set = SyntheticDataset(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TEST_DIR,
                                            label_directory=SIMULATED_LABEL_TEST_DIR)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True)

                n_epochs = 1000

                print(f"Now training on dataset with {n_biomarkers} biomarkers ({trial + 1}/{n_trials} trials)")

                # ---------- Train VAE -----------
                # Define network
                ae_enc = vae_stager.Encoder(d_in=n_biomarkers, d_latent=1).to(DEVICE)
                ae_dec = vae_stager.Decoder(d_out=n_biomarkers, d_latent=1).to(DEVICE)

                ae = vae_stager.VAE(enc=ae_enc, dec=ae_dec)

                # Define optimiser
                opt_vae = torch.optim.Adam(itertools.chain(ae_enc.parameters(), ae_dec.parameters()), lr=0.001)

                # Run training loop
                start_time = time.time()
                run_training(n_epochs, ae, dataset_name, train_loader, val_loader,
                             opt_vae, vae_stager.vae_criterion_wrapper(beta=1), model_type="vae",
                             device=DEVICE)

                # Load best model obtained from training, to run inference and evaluate.
                ae_enc_model_path = os.path.join(SAVED_MODEL_DIR, "vae", "enc_" + dataset_name + ".pth")
                ae_dec_model_path = os.path.join(SAVED_MODEL_DIR, "vae", "dec_" + dataset_name + ".pth")

                ae_enc.load_state_dict(torch.load(ae_enc_model_path, map_location=DEVICE))
                ae_dec.load_state_dict(torch.load(ae_dec_model_path, map_location=DEVICE))

                # Run inference and stop the timer
                seq_pred = infer_seq_from_network(dataloader=test_loader, net=ae).cpu()
                time_taken = time.time() - start_time  # in seconds
                time_taken_vae[-1][trial] = time_taken

                # Compute performance metrics on the test set.
                staging_rmse, _ = evaluate_autoencoder(test_loader, ae, DEVICE)

                # Read in ground truth sequence
                seq_label_file = os.path.join(SIMULATED_LABEL_TEST_DIR, dataset_name + "_seq.json")
                with open(seq_label_file, 'r') as f:
                    seq_gt = json.load(f)

                seq_gt = np.array(seq_gt).squeeze(1)  # flatten as it is a list of lists

                staging_errors_vae[-1][trial] = staging_rmse.cpu()
                sequence_scores_vae[-1][trial] = evaluate_sequence(seq_pred, seq_gt)

                # ---------- Train AE -----------
                # Define network
                ae_enc = ae_stager.Encoder(d_in=n_biomarkers, d_latent=1).to(DEVICE)
                ae_dec = ae_stager.Decoder(d_out=n_biomarkers, d_latent=1).to(DEVICE)

                ae = ae_stager.AE(enc=ae_enc, dec=ae_dec)

                # Define optimiser
                opt_ae = torch.optim.Adam(itertools.chain(ae_enc.parameters(), ae_dec.parameters()), lr=0.001)

                # Run training loop
                start_time = time.time()
                run_training(n_epochs, ae, dataset_name, train_loader, val_loader,
                             opt_ae, ae_stager.ae_criterion, model_type="ae",
                             device=DEVICE)

                # Load best model obtained from training, to run inference and evaluate.
                ae_enc_model_path = os.path.join(SAVED_MODEL_DIR, "ae", "enc_" + dataset_name + ".pth")
                ae_dec_model_path = os.path.join(SAVED_MODEL_DIR, "ae", "dec_" + dataset_name + ".pth")

                ae_enc.load_state_dict(torch.load(ae_enc_model_path, map_location=DEVICE))
                ae_dec.load_state_dict(torch.load(ae_dec_model_path, map_location=DEVICE))

                # Run inference and stop the timer
                seq_pred = infer_seq_from_network(dataloader=test_loader, net=ae).cpu()
                time_taken = time.time() - start_time  # in seconds
                time_taken_ae[-1][trial] = time_taken

                # Compute performance metrics on the test set.
                staging_rmse, _ = evaluate_autoencoder(test_loader, ae, DEVICE)

                # Read in ground truth sequence
                seq_label_file = os.path.join(SIMULATED_LABEL_TEST_DIR, dataset_name + "_seq.json")
                with open(seq_label_file, 'r') as f:
                    seq_gt = json.load(f)

                seq_gt = np.array(seq_gt).squeeze(1)  # flatten as it is a list of lists

                staging_errors_ae[-1][trial] = staging_rmse.cpu()
                sequence_scores_ae[-1][trial] = evaluate_sequence(seq_pred, seq_gt)

            # ----------- Write results to file -----------
            writer.writerow([time_taken_vae[-1].mean().item(),
                             time_taken_vae[-1].std().item(),
                             time_taken_ae[-1].mean().item(),
                             time_taken_ae[-1].std().item(),
                             staging_errors_vae[-1].mean().item(),
                             staging_errors_vae[-1].std().item(),
                             staging_errors_ae[-1].mean().item(),
                             staging_errors_ae[-1].std().item(),
                             sequence_scores_vae[-1].mean().item(),
                             sequence_scores_vae[-1].std().item(),
                             sequence_scores_ae[-1].mean().item(),
                             sequence_scores_ae[-1].std().item()])
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
    ax.plot(np.array(n_biomarkers_all), np.array(sequence_scores_vae).mean(axis=-1))
    ax.errorbar(np.array(n_biomarkers_all), np.array(sequence_scores_vae).mean(axis=-1),
                yerr=np.array(sequence_scores_vae).std(axis=-1),
                fmt='o',
                capsize=3)
    fig.suptitle("Kendall's tau score of sequence learned by VAE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Kendall's tau")
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.array(n_biomarkers_all), np.array(sequence_scores_ae).mean(axis=-1))
    ax.errorbar(np.array(n_biomarkers_all), np.array(sequence_scores_ae).mean(axis=-1),
                yerr=np.array(sequence_scores_ae).std(axis=-1),
                fmt='o',
                capsize=3)
    fig.suptitle("Kendall's tau score of sequence learned by AE")
    ax.set_xlabel("No. biomarkers")
    ax.set_ylabel("Kendall's tau")
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
