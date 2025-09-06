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
from datasets.synthetic_dataset import SyntheticDataset
from models import vae_stager, ae_stager

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    # torch.manual_seed(seed)
    print("Seed set to", seed)
    pretrain = False  # whether to use a pretrained model or not
    pretrained_model_name = "pretrained_13b"

    dataset_name = "adni_longitudinal_data"
    model_name = dataset_name

    # TRAIN MODEL ------------------------------
    # Load data
    train_dataset = SyntheticDataset(dataset_names=dataset_name, obs_directory=ADNI_DIR)

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

    # First run KDE-EBM and get the sequence and staging outputs.
    # Create a copy of the dataset formatted for use with the KDE-EBM
    prepare_csv_for_kde_ebm(os.path.join(ADNI_DIR, dataset_name + ".csv"), suffix="kde-ebm")
    # Run KDE-EBM
    ebm_sequence, _, ebm_stages = run_kde_ebm(file_dir=ADNI_DIR, file_name=dataset_name)
    ebm_stages = ebm_stages / num_biomarkers

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

        # RUN SEQUENCE INFERENCE  ------------------------------------
        biomarker_names = train_dataset.biomarker_names

        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)

        # Predict sequence
        sequence_prediction = autoencoder_sequence_inference.infer_seq_from_network(dataloader=train_loader, net=net).cpu()
        print(biomarker_names[sequence_prediction])

        # Compute Kendall's tau score between the predictions of the deep learning algorithm and KDE-EBM
        print(f"Kendall's tau between the sequences predicted by {model_type.upper()} "
              f"and KDE-EBM: {evaluate_sequence(sequence_prediction, ebm_sequence)}")

        # AUTOENCODER PLOTS ----------------------
        # Sequence visualisation
        fig, ax = plotting.plot_predicted_sequence(gt_ordering=ebm_sequence, pred_ordering=sequence_prediction,
                                                   biomarker_names=biomarker_names)
        fig.suptitle(f"Comparison between EBM (reference) and {model_type.upper()} ordering")
        fig.show()

        # todo: PVD (with y axis sorted by the the seq computed using the full dataset)

        # Predicted vs true pstages
        # fig, ax = plotting.predicted_stage_comparison(train_loader, net)
        # label = fig._suptitle.get_text()
        # fig.suptitle(label + f" ({model_type})")
        # fig.show()

        # Predicted vs EBM stages
        preds = []
        with torch.no_grad():
            for X, _ in data_loader:
                preds.append(net.predict_stage(X))

        preds = torch.concatenate(preds).squeeze().cpu()

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(ebm_stages, preds)

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

    plt.show()
