import matplotlib.pyplot as plt
import torch
import os

from config import DEVICE, PLOT_DIR, MODEL_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_VAL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_VAL_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
import ae_stager
import plotting



if __name__ == "__main__":
    dataset_name = "synthetic_120_10_dpm_0"

    # Load training and validation sets
    train_dataset = SyntheticDatasetVec(dataset_name=dataset_name, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    # val_dataset = SyntheticDatasetVec(dataset_name=dataset_name, obs_directory=SIMULATED_OBS_VAL_DIR, label_directory=SIMULATED_LABEL_VAL_DIR)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

    example_x, _ = next(iter(train_loader))
    num_biomarkers = example_x.shape[1]

    # Load model
    ae_enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    ae_dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

    ae = ae_stager.AE(enc=ae_enc, dec=ae_dec)

    enc_model_path = os.path.join(MODEL_DIR, "ae", "enc_" + dataset_name + ".pth")
    dec_model_path = os.path.join(MODEL_DIR, "ae", "dec_" + dataset_name + ".pth")

    ae_enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
    ae_dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

    # Plot biomarker levels against predicted stages
    fig, ax = plotting.staged_biomarker_plots(train_loader, num_biomarkers, ae, DEVICE)
    fig.show()

    # Plot predicted stages against true stages
    fit, ax = plotting.predicted_stage_comparison(train_loader, num_biomarkers, ae, DEVICE)

    plt.show()
