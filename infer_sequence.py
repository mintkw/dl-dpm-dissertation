import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_VAL_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_VAL_DIR
from models import ae_stager, vae_stager
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from evaluation import evaluate_autoencoder, evaluate_sequence
import stages_to_sequence


if __name__ == "__main__":
    # USER CONFIGURATION --------------------
    # dataset_names = [f"synthetic_120_10_{i}" for i in range(2)]
    dataset_names = "synthetic_120_10_1"
    model_name = "synthetic_120_10_0"

    model_type = "vae"  # only vae or ae supported currently
    if model_type not in ["vae", "ae"]:
        print("Model type must be one of 'vae' or 'ae' (case-sensitive)")
        exit()
    # ---------------------------------------

    # Load datasets
    train_dataset = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_VAL_DIR, label_directory=SIMULATED_LABEL_VAL_DIR)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

    example_x, _ = next(iter(train_loader))
    num_biomarkers = example_x.shape[1]

    # Read in ground truth sequence - note that the val and train sets share one sequence
    seq_label_file = os.path.join(SIMULATED_LABEL_TRAIN_DIR, dataset_names + "_seq.json")
    with open(seq_label_file, 'r') as f:
        seq_gt = json.load(f)

    # flatten sequence
    seq_gt = np.array(seq_gt).squeeze(1)

    print("gt sequence:\n", seq_gt)

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

    # Verify that everything loaded alright by printing mean squared error on training set
    stage_mse_train, reconst_mse_train = evaluate_autoencoder(train_loader, net, DEVICE)
    stage_mse_val, reconst_mse_val = evaluate_autoencoder(val_loader, net, DEVICE)

    print("Reconstruction MSE on training set:", reconst_mse_train)
    print("Reconstruction MSE on validation set:", reconst_mse_val)
    print()
    print("Staging MSE on training set:", stage_mse_train)
    print("Staging MSE on validation set:", stage_mse_val)

    # Infer sequence from stage information
    seq_prediction = stages_to_sequence.stages_to_sequence_direct(num_biomarkers, train_loader, net, DEVICE)

    print("seq estimated directly from predictions:")
    print(seq_prediction)
    print("directly-estimated sequence score:", evaluate_sequence(seq_prediction.cpu(), seq_gt))

    # Infer sequence by fitting curves to each biomarker against pseudo-time
    # Fit biomarker curves
    # sigmoid_params_train, fig_train, ax_train = stages_to_sequence.fit_biomarker_curves(train_loader, net, n_epochs=500, device=DEVICE, lr=0.01)
    # label = fig_train._suptitle.get_text()
    # fig_train.suptitle(label + " on training set")
    # fig_train.show()

    sigmoid_params_val, fig_val, ax_val = stages_to_sequence.fit_biomarker_curves(val_loader, net, n_epochs=500, device=DEVICE, lr=0.01)
    label = fig_val._suptitle.get_text()
    fig_val.suptitle(label + " on validation set")
    fig_val.show()

    # # Infer sequence from the fitted curves
    # curve_fitting_seq_train = stages_to_sequence.infer_seq_from_biomarker_curves(sigmoid_params_train)
    # print("seq inferred from fitted biomarker curves on training set:")
    # print(curve_fitting_seq_train)
    # print("Curve-fitted sequence score:", evaluate_sequence(curve_fitting_seq_train.cpu(), seq_gt))

    curve_fitting_seq_val = stages_to_sequence.infer_seq_from_biomarker_curves(sigmoid_params_val)
    print("seq inferred from fitted biomarker curves on validation set:")
    print(curve_fitting_seq_val)
    print("Curve-fitted sequence score:", evaluate_sequence(curve_fitting_seq_val.cpu(), seq_gt))

    plt.show()
