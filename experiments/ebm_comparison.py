import json

import torch
import itertools
import time
import csv
import numpy as np
import os

from sklearn.model_selection import train_test_split

from datasets.simulate_data import generate_data
from datasets.kde_ebm_dataset_prep import prepare_csv_for_kde_ebm
from dpm_algorithms.run_kde_ebm import run_kde_ebm
from dpm_algorithms.train_autoencoder import run_training
from datasets.synthetic_dataset import SyntheticDataset
from models import ae_stager, vae_stager
from dpm_algorithms.evaluation import evaluate_autoencoder, evaluate_sequence
from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_TEST_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_TEST_DIR
from dpm_algorithms.autoencoder_sequence_inference import infer_seq_from_network


if __name__ == "__main__":
    n_biomarkers_all = [10, 50, 100, 250]

    time_taken_ae = []
    time_taken_vae = []
    time_taken_ebm = []
    sequence_scores_ae = []
    sequence_scores_vae = []
    sequence_scores_ebm = []

    for n_biomarkers in n_biomarkers_all:
        print(f"Now running on dataset with {n_biomarkers} biomarkers")

        # Generate data
        generate_data(n_biomarkers=n_biomarkers, n_mci=10 * n_biomarkers,
                      n_controls=n_biomarkers, n_patients=n_biomarkers,
                      num_sets=1)

        # Load dataset
        dataset_name = f"synthetic_{12 * n_biomarkers}_{n_biomarkers}_0"

        # Create a copy of the dataset formatted for use with the KDE-EBM
        prepare_csv_for_kde_ebm(os.path.join(SIMULATED_OBS_TRAIN_DIR, dataset_name + ".csv"), suffix="kde-ebm")

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

        n_epochs = 1000

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
        seq_pred = infer_seq_from_network(dataloader=train_loader, net=ae).cpu()
        time_taken = time.time() - start_time  # in seconds
        time_taken_vae.append(time_taken)

        # Read in ground truth sequence
        seq_label_file = os.path.join(SIMULATED_LABEL_TEST_DIR, dataset_name + "_seq.json")
        with open(seq_label_file, 'r') as f:
            seq_gt = json.load(f)

        seq_gt = np.array(seq_gt).squeeze(1)  # flatten as it is a list of lists

        # Evaluate the predicted sequence
        sequence_scores_vae.append(evaluate_sequence(seq_pred, seq_gt))

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
        seq_pred = infer_seq_from_network(dataloader=train_loader, net=ae).cpu()
        time_taken = time.time() - start_time  # in seconds
        time_taken_ae.append(time_taken)

        # Read in ground truth sequence
        seq_label_file = os.path.join(SIMULATED_LABEL_TEST_DIR, dataset_name + "_seq.json")
        with open(seq_label_file, 'r') as f:
            seq_gt = json.load(f)

        seq_gt = np.array(seq_gt).squeeze(1)  # flatten as it is a list of lists

        # Evaluate the predicted sequence
        sequence_scores_ae.append(evaluate_sequence(seq_pred, seq_gt))

        # ------------ Run KDE-EBM ---------------
        ml_order, time_taken = run_kde_ebm(file_dir=SIMULATED_OBS_TRAIN_DIR, file_name=dataset_name)
        time_taken_ebm.append(time_taken)

        # Evaluate the predicted sequence
        sequence_scores_ebm.append(evaluate_sequence(ml_order, seq_gt))

    # Write out results to file
    with open('ebm_comparison.csv', 'w', newline='') as output_f:
        writer = csv.writer(output_f)
        writer.writerow(["vae_time", "ae_time", "ebm_time", "vae_kt_score", "ae_kt_score", "ebm_kt_score"])

        writer.writerows([[time_taken_vae[i], time_taken_ae[i], time_taken_ebm[i],
                           sequence_scores_vae[i], sequence_scores_ae[i], sequence_scores_ebm[i]]
                          for i in range(len(n_biomarkers_all))])
