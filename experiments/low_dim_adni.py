import json
import os

import numpy as np
import torch
import itertools

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from dpm_algorithms import plotting, autoencoder_sequence_inference, train_autoencoder
from dpm_algorithms.evaluation import evaluate_autoencoder, evaluate_sequence
from dpm_algorithms.run_kde_ebm import run_kde_ebm
from config import DEVICE, SAVED_MODEL_DIR, ADNI_DIR
from datasets.kde_ebm_dataset_prep import prepare_csv_for_kde_ebm
from datasets.biomarker_dataset import BiomarkerDataset
from models import vae_stager, ae_stager

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set to", seed)
    pretrain = False  # whether to use a pretrained model or not
    pretrained_model_name = "pretrained_13b"
    n_trials = 1  # Set the number of times to repeat the experiment.

    dataset_name = "adni_summed_volumes_abpos"
    model_name = dataset_name

    # TRAIN MODEL ------------------------------
    # Load data
    train_dataset = BiomarkerDataset(dataset_names=dataset_name, obs_directory=ADNI_DIR)

    # Split training set
    train_indices, val_indices = train_test_split(range(len(train_dataset)), train_size=0.8, random_state=seed)

    # Generate subset based on indices
    train_split = torch.utils.data.Subset(train_dataset, train_indices)
    val_split = torch.utils.data.Subset(train_dataset, val_indices)

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=8, shuffle=True)

    num_biomarkers = next(iter(train_loader))[0].shape[1]

    n_epochs = 1000

    model_types = ["ae", "vae"]
    reconstruction_errors = [[] for _ in range(len(model_types))]
    staging_errors = [[] for _ in range(len(model_types))]
    sequence_scores = [[] for _ in range(len(model_types))]

    for trial in range(n_trials):
        enc = None
        dec = None

        # First run KDE-EBM and get the sequence and staging outputs.
        # Create a copy of the dataset formatted for use with the KDE-EBM
        prepare_csv_for_kde_ebm(os.path.join(ADNI_DIR, dataset_name + ".csv"), suffix="kde-ebm")
        # Run KDE-EBM
        ebm_sequence, _, ebm_stages = run_kde_ebm(file_dir=ADNI_DIR, file_name=dataset_name, greedy_n_init=10)

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
            # train_autoencoder.run_training(n_epochs, net, model_name, train_loader, val_loader, opt,
            #                                criterion, model_type=model_type, device=DEVICE, min_epochs=50)

            # Load the best models saved during training
            enc_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "enc_" + model_name + ".pth")
            dec_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "dec_" + model_name + ".pth")

            enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
            dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

            data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)

            # Compute reconstruction error
            _, reconstruction_mse = evaluate_autoencoder(data_loader, net, DEVICE)

            # Compute RMS between the autoencoder and EBM stages
            preds = []
            diagnoses = []
            with torch.no_grad():
                for X, diagnosis in data_loader:
                    preds.append(net.predict_stage(X))
                    diagnoses.append(diagnosis.cpu())

            preds = torch.concatenate(preds).squeeze().cpu()
            diagnoses = torch.concatenate(diagnoses).squeeze().cpu()
            staging_rms = torch.sqrt(torch.mean(((preds * num_biomarkers) - torch.tensor(ebm_stages)) ** 2))

            # RUN SEQUENCE INFERENCE  ------------------------------------
            biomarker_names = train_dataset.biomarker_names

            # Predict sequence
            sequence_prediction = autoencoder_sequence_inference.infer_seq_from_network(dataloader=data_loader, net=net).cpu()
            print(biomarker_names[sequence_prediction])

            # Compute Kendall's tau score between the predictions of the deep learning algorithm and KDE-EBM
            seq_score = evaluate_sequence(sequence_prediction, ebm_sequence)
            print(f"Kendall's tau between the sequences predicted by {model_type.upper()} "
                  f"and KDE-EBM: {seq_score}")

            reconstruction_errors[i].append(reconstruction_mse)
            staging_errors[i].append(staging_rms)
            sequence_scores[i].append(seq_score)

            # AUTOENCODER PLOTS ----------------------
            if trial == n_trials - 1:
                # Only compute plots for the last experiment run.
                # Sequence visualisation
                fig, ax = plotting.plot_predicted_sequence(gt_ordering=ebm_sequence, pred_ordering=sequence_prediction,
                                                           biomarker_names=biomarker_names)
                fig.suptitle(f"Comparison between EBM (reference) and {model_type.upper()} ordering")
                fig.show()

                # PVD (with y axis sorted by the seq computed by KDE-EBM)
                fig, ax = plotting.positional_var_diagram(gt_ordering=ebm_sequence, dataset=train_dataset, net=net,
                                                          biomarker_names=biomarker_names)
                label = fig._suptitle.get_text()
                fig.suptitle(label + f" ({model_type})")
                fig.show()

                # Histogram of inferred pseudo-stage grouped by true diagnosis
                # fig, ax = plotting.predicted_stage_comparison(train_loader, net)
                # label = fig._suptitle.get_text()
                # fig.suptitle(label + f" ({model_type})")
                # fig.show()
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
                hist_dat = [preds[diagnoses == 0],
                            preds[diagnoses == 1],
                            preds[diagnoses == 2]]

                axes[0].boxplot(hist_dat)
                axes[0].set_xticklabels(['CN', 'MCI', 'AD'], rotation=0)
                axes[0].set_ylabel("Pseudo-stage inferred by the autoencoder")
                axes[0].set_xlabel("Clinical diagnosis")
                fig.suptitle("Distribution of pseudo-stage inferences for data of each diagnosis")

                leg1 = axes[1].hist(hist_dat,
                               density=True,
                               alpha=0.5,
                               stacked=True)
                axes[1].set_xlabel("Pseudo-stage inferred by the autoencoder")

                fig.legend(leg1[2], ['CN', 'AD', 'MCI'],
                           bbox_to_anchor=(1, 1), loc="upper right", fontsize=15)
                fig.tight_layout()

                fig.show()

                # Predicted vs EBM stages
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.scatter(ebm_stages / num_biomarkers, preds)

                # Plot a straight line for comparison
                ax.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2))

                fig.suptitle(f"{model_type.upper()} stages vs EBM stages")
                ax.set_xlabel("EBM stage")
                ax.set_ylabel(f"{model_type.upper()} stage")
                fig.show()

                # Grid of inferred trajectories using the encoder (predicted pstage and biomarker data)
                fig, ax = plotting.encoder_trajectories_estimate(train_loader, net, biomarker_names=biomarker_names)
                label = fig._suptitle.get_text()
                fig.suptitle(label + f" ({model_type.upper()})")
                fig.show()

                # Grid of inferred trajectories using the decoder (mapping from latent space)
                fig, ax = plotting.decoder_trajectories_estimate(num_biomarkers, net, biomarker_names=biomarker_names)
                label = fig._suptitle.get_text()
                fig.suptitle(label + f" ({model_type.upper()})")
                fig.show()

                # Overlaid inferred trajectories using the encoder
                fig, ax = plotting.encoder_progression_estimate(train_loader, net, normalise=False,
                                                                biomarker_names=biomarker_names)
                label = fig._suptitle.get_text()
                fig.suptitle(label + f" ({model_type.upper()})")
                fig.show()

                # Overlaid inferred trajectories using the decoder
                fig, ax = plotting.decoder_progression_estimate(train_loader, net, biomarker_names=biomarker_names)
                label = fig._suptitle.get_text()
                fig.suptitle(label + f" ({model_type.upper()})")
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
