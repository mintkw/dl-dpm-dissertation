import json
import os

import numpy as np
import torch
import itertools

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from dpm_algorithms import plotting, train_autoencoder
from dpm_algorithms.autoencoder_sequence_inference import infer_seq_from_network
from config import SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TRAIN_DIR, DEVICE, SAVED_MODEL_DIR, \
    SIMULATED_LABEL_TEST_DIR, SIMULATED_OBS_TEST_DIR
from datasets import simulate_data
from datasets.biomarker_dataset import BiomarkerDataset
from dpm_algorithms.evaluation import evaluate_autoencoder, evaluate_sequence
from models import vae_stager, ae_stager

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set to", seed)
    n_biomarkers = 10
    pretrain = True  # whether to use a pretrained model or not
    pretrained_model_name = f"pretrained_{n_biomarkers}bm"  # run pretrain_a_model with the expected number of biomarkers first.
    n_trials = 1  # Set the number of times to repeat the experiment.

    model_types = ["ae", "vae"]
    reconstruction_errors = [[] for _ in range(len(model_types))]
    staging_errors = [[] for _ in range(len(model_types))]
    sequence_scores = [[] for _ in range(len(model_types))]

    for trial in range(n_trials):
        # GENERATE THE DATA --------------------------
        n_mci = 100
        n_controls = 10
        n_patients = 10
        num_sets = 1
        file_name = "iexample"

        seq = np.array([4, 5, 9, 0, 7, 6, 8, 2, 3, 1])[:, None].tolist()

        simulate_data.generate_data(n_biomarkers=n_biomarkers, n_mci=n_mci, n_controls=n_controls, n_patients=n_patients,
                                    plot=False, num_sets=num_sets, file_name=file_name, seq=seq)

        # TRAIN MODEL ------------------------------
        # due to the convention of generate_data to append an index to each dataset in a batch, we append "_0"
        dataset_name = file_name + "_0"
        model_name = file_name + "_0"

        # Load data
        train_dataset = BiomarkerDataset(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TRAIN_DIR,
                                         label_directory=SIMULATED_LABEL_TRAIN_DIR)

        # Split training set
        train_indices, val_indices = train_test_split(range(len(train_dataset)), train_size=0.8, random_state=seed)

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

        for i in range(len(model_types)):
            model_type = model_types[i]

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
            test_dataset = BiomarkerDataset(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TEST_DIR,
                                            label_directory=SIMULATED_LABEL_TEST_DIR)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

            # INFER SEQUENCE FROM THE TEST SET ------------------
            # Read in ground truth sequence
            seq_label_file = os.path.join(SIMULATED_LABEL_TEST_DIR, dataset_name + "_seq.json")
            with open(seq_label_file, 'r') as f:
                seq_gt = json.load(f)

            seq_gt = np.array(seq_gt).squeeze(1)  # flatten as it is a list of lists

            # Predict sequence
            sequence_prediction = infer_seq_from_network(dataloader=test_loader, net=net).cpu()
            print(sequence_prediction)

            # COMPUTE EVALUATION METRICS --------------
            staging_rmse, reconstruction_mse = evaluate_autoencoder(test_loader, net, DEVICE)
            # reconstruction_mse = reconstruction_mse.cpu()
            sequence_score = evaluate_sequence(sequence_prediction, seq_gt)

            print(f"Staging RMSE of {model_type} on test set: {staging_rmse}")
            print(f"Reconstruction MSE of {model_type} on test set: {reconstruction_mse}")
            print(f"Score of predicted sequence: {sequence_score}")

            staging_errors[i].append(staging_rmse)
            reconstruction_errors[i].append(reconstruction_mse)
            sequence_scores[i].append(sequence_score)

            # PLOTS ----------------------
            if trial == n_trials - 1:
                # Only compute plots for the last experiment run.
                biomarker_names = test_dataset.biomarker_names

                # Sequence visualisation
                fig, ax = plotting.plot_predicted_sequence(gt_ordering=seq_gt, pred_ordering=sequence_prediction)
                label = fig._suptitle.get_text()
                fig.suptitle(label + f" ({model_type})")
                fig.show()

                # PVD (with y axis sorted by the seq computed using the full dataset)
                fig, ax = plotting.positional_var_diagram(gt_ordering=seq_gt, dataset=test_dataset, net=net)
                label = fig._suptitle.get_text()
                fig.suptitle(label + f" ({model_type})")
                fig.show()

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
                fig, ax = plotting.encoder_progression_estimate(train_loader, net, normalise=False,
                                                                biomarker_names=biomarker_names)
                label = fig._suptitle.get_text()
                fig.suptitle(label + f" ({model_type})")
                fig.show()

                # Overlaid inferred trajectories using the decoder
                fig, ax = plotting.decoder_progression_estimate(train_loader, net, biomarker_names=biomarker_names)
                label = fig._suptitle.get_text()
                fig.suptitle(label + f" ({model_type})")
                fig.show()

    # Compute mean and stdev of the evaluation metrics over all trials
    staging_errors = torch.tensor(staging_errors)
    reconstruction_errors = torch.tensor(reconstruction_errors)
    sequence_scores = torch.tensor(sequence_scores)

    for i in range(len(model_types)):
        print(f"Staging RMSE of {model_types[i]}: {staging_errors[i].mean()} +- {staging_errors[i].std()}")
        print(f"Reconstruction MSE of {model_types[i]}: {reconstruction_errors[i].mean()} +- {reconstruction_errors[i].std()}")
        print(f"Sequence Kendall's tau score of {model_types[i]}: {sequence_scores[i].mean()} +- {sequence_scores[i].std()}")
        print()

    plt.show()
