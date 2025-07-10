import matplotlib.pyplot as plt
import torch
import os

from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_VAL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_VAL_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from models import ae_stager, vae_stager
import plotting


if __name__ == "__main__":
    # USER CONFIGURATION --------------------
    # dataset_name = "synthetic_120_10_dpm_same"
    dataset_name = "synthetic_600_50_0"

    model_type = "ae"  # only vae or ae supported currently
    if model_type not in ["vae", "ae"]:
        print("Model type must be one of 'vae' or 'ae' (case-sensitive)")
        exit()
    # ---------------------------------------

    # Load training and validation sets
    train_dataset = SyntheticDatasetVec(dataset_name=dataset_name, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataset = SyntheticDatasetVec(dataset_name=dataset_name, obs_directory=SIMULATED_OBS_VAL_DIR, label_directory=SIMULATED_LABEL_VAL_DIR)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

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
    enc_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "enc_" + dataset_name + ".pth")
    dec_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "dec_" + dataset_name + ".pth")

    enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
    dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

    # Plot biomarker levels against predicted stages
    fig, ax = plotting.staged_biomarker_plots(train_loader, net, DEVICE)
    label = fig._suptitle.get_text()
    fig.suptitle(label + " on training set")
    fig.show()

    fig, ax = plotting.staged_biomarker_plots(val_loader, net, DEVICE)
    label = fig._suptitle.get_text()
    fig.suptitle(label + " on validation set")
    fig.show()

    # Plot predicted stages against true stages
    fig, ax = plotting.predicted_stage_comparison(train_loader, num_biomarkers, net, DEVICE)
    label = fig._suptitle.get_text()
    fig.suptitle(label + " on training set")
    fig.show()

    fig, ax = plotting.predicted_stage_comparison(val_loader, num_biomarkers, net, DEVICE)
    label = fig._suptitle.get_text()
    fig.suptitle(label + " on validation set")
    fig.show()

    # Plot predicted biomarker trajectories
    fig, ax = plotting.predicted_biomarker_trajectories(num_biomarkers, net, DEVICE)
    fig.show()

    plt.show()
