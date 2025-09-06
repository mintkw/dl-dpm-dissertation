import os

import numpy as np
import torch
import itertools

from sklearn.model_selection import train_test_split

from config import SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TRAIN_DIR, DEVICE, SAVED_MODEL_DIR, \
    SIMULATED_LABEL_TEST_DIR, SIMULATED_OBS_TEST_DIR
from datasets import simulate_data
from dpm_algorithms import train_autoencoder
from datasets.synthetic_dataset import SyntheticDataset
from dpm_algorithms.evaluation import evaluate_autoencoder
from models import vae_stager, ae_stager


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set to", seed)

    # GENERATE THE DATA --------------------------
    num_sets = 50

    # ideal datasets
    n_biomarkers = 10
    n_mci = 100
    n_controls = 10
    n_patients = 10
    file_name = "ideal_example"

    # onsets_stages = np.sort(np.random.rand(n_biomarkers))  # randomly determine onset times

    simulate_data.generate_data(n_biomarkers=n_biomarkers, n_mci=n_mci, n_controls=n_controls, n_patients=n_patients,
                                plot=False, num_sets=num_sets, file_name=file_name)

    # realistic datasets
    n_mci = 1000
    n_controls = 100
    n_patients = 100
    file_name = "noisy_example"

    means_normal = np.random.normal(loc=-0.5, scale=0.2, size=n_biomarkers)
    # compute in this way to force increasing trajectories
    means_abnormal = means_normal + (np.random.normal(loc=1, scale=0.2, size=n_biomarkers) ** 2)

    sds_normal = np.random.rand(n_biomarkers) * 0.5
    sds_abnormal = np.random.rand(n_biomarkers) * 0.5

    simulate_data.generate_data(n_biomarkers=n_biomarkers, n_mci=n_mci, n_controls=n_controls, n_patients=n_patients,
                                means_normal=means_normal, means_abnormal=means_abnormal,
                                sds_normal=sds_normal, sds_abnormal=sds_abnormal, plot=False,
                                num_sets=num_sets, file_name=file_name)

    # TRAIN MODEL ------------------------------
    dataset_names = [f"ideal_example_{i}" for i in range(num_sets)] + [f"noisy_example_{i}" for i in range(num_sets)]
    model_name = f"pretrained_{n_biomarkers}bm"

    # Load data
    train_dataset = SyntheticDataset(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TRAIN_DIR,
                                     label_directory=SIMULATED_LABEL_TRAIN_DIR)

    # Split training set
    train_indices, val_indices = train_test_split(range(len(train_dataset)), train_size=0.8)

    # Generate subset based on indices
    train_split = torch.utils.data.Subset(train_dataset, train_indices)
    val_split = torch.utils.data.Subset(train_dataset, val_indices)

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=16, shuffle=True)

    num_biomarkers = next(iter(train_loader))[0].shape[1]

    n_epochs = 1000

    enc = None
    dec = None

    for model_type in ["ae", "vae"]:
        # Instantiate network
        if model_type == "vae":
            enc = vae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
            dec = vae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

            net = vae_stager.VAE(enc=enc, dec=dec)
            criterion = vae_stager.vae_criterion_wrapper(beta=1)

        elif model_type == "ae":
            enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
            dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

            net = ae_stager.AE(enc=enc, dec=dec)
            criterion = ae_stager.ae_criterion

        opt = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=0.0001)

        # Run training loop
        train_autoencoder.run_training(n_epochs, net, model_name, train_loader, val_loader, opt,
                                       criterion, model_type=model_type, device=DEVICE)

        # Load the best models saved during training
        enc_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "enc_" + model_name + ".pth")
        dec_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "dec_" + model_name + ".pth")

        enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
        dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

        # EVALUATION ON TEST SET ------------------------------------
        test_dataset = SyntheticDataset(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TEST_DIR,
                                        label_directory=SIMULATED_LABEL_TEST_DIR)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

        staging_rmse, reconstruction_mse = evaluate_autoencoder(test_loader, net, DEVICE)
        reconstruction_mse = reconstruction_mse.cpu()

        print(f"Staging RMSE of {model_type} on test set: {staging_rmse}")
        print(f"Reconstruction MSE of {model_type} on test set: {reconstruction_mse}")

