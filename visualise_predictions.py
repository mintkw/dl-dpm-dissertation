import matplotlib.pyplot as plt
import torch
import os

from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_OBS_TRAIN_DIR, ADNI_DIR
from datasets.synthetic_dataset import SyntheticDataset
from models import ae_stager, vae_stager
from dpm_algorithms import plotting

if __name__ == "__main__":
    # USER CONFIGURATION --------------------
    # num_sets = 1
    # dataset_names = [f"synthetic_120_10_{i}" for i in range(num_sets)]
    # model_name = "synthetic_120_10_0"
    # dataset_names = ["adni_longitudinal_data_all_volumes_ABpos"]
    # dataset_names = ["example_0"]
    dataset_names = ["synthetic_1200_100_0"]
    model_name = dataset_names[0]

    model_type = "vae"  # only vae or ae supported currently
    if model_type not in ["vae", "ae"]:
        print("Model type must be one of 'vae' or 'ae' (case-sensitive)")
        exit()

    dataset_type = "synthetic"  # only 'synthetic' or 'adni' supported currently
    if dataset_type not in ["synthetic", "adni"]:
        print("Dataset type must be one of 'synthetic' or 'adni' (case-sensitive)")
        exit()
    # ---------------------------------------

    # Load training set
    if dataset_type == "synthetic":
        train_dataset = SyntheticDataset(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    elif dataset_type == "adni":
        train_dataset = SyntheticDataset(dataset_names=dataset_names, obs_directory=ADNI_DIR)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    biomarker_names = train_dataset.biomarker_names

    example_x, _ = next(iter(train_loader))
    num_biomarkers = example_x.shape[1]

    # Instantiate saved_models
    enc = None
    dec = None

    if model_type == "vae":
        enc = vae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
        dec = vae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

        net = vae_stager.VAE(enc=enc, dec=dec)

    elif model_type == "ae":
        enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
        dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

        net = ae_stager.AE(enc=enc, dec=dec)

    if enc is None or dec is None:
        print("Arguments not validated properly - please fix")
        exit()

    # Load a model fitted to the particular dataset
    enc_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "enc_" + model_name + ".pth")
    dec_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "dec_" + model_name + ".pth")

    enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
    dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

    # Plot average biomarker value for each stage prediction
    fig, ax = plotting.encoder_progression_estimate(train_loader, net, normalise=True, biomarker_names=biomarker_names)
    label = fig._suptitle.get_text()
    fig.suptitle(label + " on training set")
    fig.show()

    # Plot biomarker levels against predicted stages
    fig, ax = plotting.encoder_trajectories_estimate(train_loader, net, biomarker_names=biomarker_names)
    label = fig._suptitle.get_text()
    fig.suptitle(label + " on training set")
    fig.show()

    # Plot predicted biomarker trajectories
    fig, ax = plotting.decoder_trajectories_estimate(num_biomarkers, net, biomarker_names=biomarker_names)
    label = fig._suptitle.get_text()
    fig.suptitle(label + " on training set")
    fig.show()

    # This check is because the ADNI data using has no 'true' stage labels.
    if dataset_type == "synthetic":
        # Plot ground truth biomarker trajectories
        fig, ax = plotting.true_trajectories(train_loader)
        label = fig._suptitle.get_text()
        fig.suptitle(label + " on training set")
        fig.show()

        # Plot predicted stages against true stages
        fig, ax = plotting.predicted_stage_comparison(train_loader, net)
        label = fig._suptitle.get_text()
        fig.suptitle(label + " on training set")
        fig.show()

    # Overlaid inferred trajectories using normalised decoder outputs
    fig, ax = plotting.decoder_progression_estimate(train_loader, net, biomarker_names=biomarker_names)
    label = fig._suptitle.get_text()
    fig.suptitle(label + " on training set")
    fig.show()

    plt.show()
