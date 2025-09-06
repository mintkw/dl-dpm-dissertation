import json
import os

import numpy as np
import torch
import itertools

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from dpm_algorithms import plotting, autoencoder_sequence_inference, train_autoencoder
from config import SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TRAIN_DIR, DEVICE, SAVED_MODEL_DIR, \
    SIMULATED_LABEL_TEST_DIR, SIMULATED_OBS_TEST_DIR
from datasets import simulate_data
from datasets.synthetic_dataset import SyntheticDataset
from dpm_algorithms.evaluation import evaluate_autoencoder, evaluate_sequence
from models import vae_stager, ae_stager


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    # torch.manual_seed(seed)
    print("Seed set to", seed)
    pretrain = False  # whether to use a pretrained model or not
    pretrained_model_name = "pretrained_10bm"  # run pretrain_a_model with the expected number of biomarkers first

    # GENERATE THE DATA --------------------------
    n_biomarkers = 10
    n_mci = 1000
    n_controls = 100
    n_patients = 100
    num_sets = 1
    file_name = "rexample"

    means_normal = np.random.normal(loc=-0.5, scale=0.1, size=n_biomarkers)
    # compute in this way to force increasing trajectories
    means_abnormal = means_normal + (np.random.normal(loc=1, scale=0.1, size=n_biomarkers) ** 2)

    sds_normal = np.ones(n_biomarkers) * 0.5
    sds_abnormal = np.ones(n_biomarkers) * 0.5

    # onsets_stages = np.sort(np.random.rand(n_biomarkers))  # randomly determine onset times

    simulate_data.generate_data(n_biomarkers=n_biomarkers, n_mci=n_mci, n_controls=n_controls, n_patients=n_patients,
                                means_normal=means_normal, means_abnormal=means_abnormal,
                                sds_normal=sds_normal, sds_abnormal=sds_abnormal, plot=False,
                                num_sets=num_sets, file_name=file_name)

    # TRAIN MODEL ------------------------------
    # due to the convention of generate_data to append an index to each dataset in a batch, we append "_0"
    dataset_name = file_name + "_0"
    model_name = file_name + "_0"

    # Load data
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

        if pretrain:
            enc_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "enc_" + pretrained_model_name + ".pth")
            dec_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "dec_" + pretrained_model_name + ".pth")

            enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
            dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

        opt = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=0.001)

        # Run training loop
        train_autoencoder.run_training(n_epochs, net, model_name, train_loader, val_loader, opt,
                                       criterion, model_type=model_type, device=DEVICE)

        # Load the best models saved during training
        enc_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "enc_" + model_name + ".pth")
        dec_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "dec_" + model_name + ".pth")

        enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
        dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

        # EVALUATION ON TEST SET ------------------------------------
        test_dataset = SyntheticDataset(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TEST_DIR,
                                        label_directory=SIMULATED_LABEL_TEST_DIR)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

        # INFER SEQUENCE FROM THE TEST SET ------------------
        # Read in ground truth sequence
        seq_label_file = os.path.join(SIMULATED_LABEL_TEST_DIR, dataset_name + "_seq.json")
        with open(seq_label_file, 'r') as f:
            seq_gt = json.load(f)

        seq_gt = np.array(seq_gt).squeeze(1)  # flatten as it is a list of lists

        # Predict sequence
        sequence_prediction = autoencoder_sequence_inference.infer_seq_from_network(dataloader=test_loader, net=net).cpu()
        print(sequence_prediction)

        # COMPUTE EVALUATION METRICS --------------
        staging_rmse, reconstruction_mse = evaluate_autoencoder(test_loader, net, DEVICE)
        reconstruction_mse = reconstruction_mse.cpu()

        print(f"Staging RMSE of {model_type} on test set: {staging_rmse}")
        print(f"Reconstruction MSE of {model_type} on test set: {reconstruction_mse}")

        print("Score of predicted sequence:", evaluate_sequence(sequence_prediction, seq_gt))

        # PLOTS ----------------------
        biomarker_names = test_dataset.biomarker_names

        # Sequence visualisation
        fig, ax = plotting.plot_predicted_sequence(gt_ordering=seq_gt, pred_ordering=sequence_prediction)
        label = fig._suptitle.get_text()
        fig.suptitle(label + f" ({model_type})")
        fig.show()

        # todo: PVD (with y axis sorted by the the seq computed using the full dataset)

        # Predicted vs true pstages
        fig, ax = plotting.predicted_stage_comparison(train_loader, net)
        label = fig._suptitle.get_text()
        fig.suptitle(label + f" ({model_type})")
        fig.show()

        # Grid of inferred trajectories using the encoder (predicted pstage and biomarker data)
        fig, ax = plotting.encoder_trajectories_estimate(train_loader, net, biomarker_names=biomarker_names)
        label = fig._suptitle.get_text()
        fig.suptitle(label + f" ({model_type})")
        fig.show()

        # Grid of inferred trajectories using the decoder (mapping from latent space)
        fig, ax = plotting.decoder_trajectories_estimate(num_biomarkers, net, biomarker_names=biomarker_names)
        label = fig._suptitle.get_text()
        fig.suptitle(label + f" ({model_type})")
        fig.show()

        # Overlaid inferred trajectories using the encoder
        fig, ax = plotting.encoder_progression_estimate(train_loader, net, normalise=True,
                                                        biomarker_names=biomarker_names)
        label = fig._suptitle.get_text()
        fig.suptitle(label + f" ({model_type})")
        fig.show()

        # Overlaid inferred trajectories using the decoder
        fig, ax = plotting.decoder_progression_estimate(train_loader, net, biomarker_names=biomarker_names)
        label = fig._suptitle.get_text()
        fig.suptitle(label + f" ({model_type})")
        fig.show()

    plt.show()
