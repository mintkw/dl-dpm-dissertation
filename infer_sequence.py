import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import plotting
from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_TEST_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_TEST_DIR
from models import ae_stager, vae_stager
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from evaluation import evaluate_autoencoder, evaluate_sequence
import stages_to_sequence


if __name__ == "__main__":
    # USER CONFIGURATION --------------------
    # dataset_names = [f"synthetic_120_10_{i}" for i in range(2)]
    dataset_name = "synthetic_1200_100_0"
    model_name = dataset_name

    model_type = "vae"  # only vae or ae supported currently
    if model_type not in ["vae", "ae"]:
        print("Model type must be one of 'vae' or 'ae' (case-sensitive)")
        exit()
    # ---------------------------------------

    # Load datasets
    train_dataset = SyntheticDatasetVec(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = SyntheticDatasetVec(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TEST_DIR, label_directory=SIMULATED_LABEL_TEST_DIR)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

    example_x, _ = next(iter(train_loader))
    num_biomarkers = example_x.shape[1]

    # Read in ground truth sequence - note that the val and train sets share one sequence
    seq_label_file = os.path.join(SIMULATED_LABEL_TRAIN_DIR, dataset_name + "_seq.json")
    with open(seq_label_file, 'r') as f:
        seq_gt = json.load(f)

    # flatten sequence
    seq_gt = np.array(seq_gt).squeeze(1)

    # print("gt sequence:\n", seq_gt)

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

    # # Infer sequence from stage information
    # seq_prediction = stages_to_sequence.stages_to_sequence_direct(num_biomarkers, train_loader, net, DEVICE)
    #
    # print("seq estimated directly from predictions:")
    # print(seq_prediction)
    # print("directly-estimated sequence score:", evaluate_sequence(seq_prediction.cpu(), seq_gt))
    #
    # # Infer sequence by fitting curves to each biomarker against pseudo-time ----------------------
    # # Plot fitted curves
    # sigmoid_params_val, fig_val, ax_val = stages_to_sequence.fit_biomarker_curves(val_loader, net, n_epochs=500, device=DEVICE, lr=0.01)
    # label = fig_val._suptitle.get_text()
    # fig_val.suptitle(label + " on validation set")
    # fig_val.show()
    #
    # # Compute score of the sequence obtained from the fitted curves
    # curve_fitting_seq_val = stages_to_sequence.infer_seq_from_biomarker_curves(sigmoid_params_val)
    # print("seq inferred from fitted biomarker curves on validation set:")
    # print(curve_fitting_seq_val)
    # print("Curve-fitted sequence score:", evaluate_sequence(curve_fitting_seq_val.cpu(), seq_gt))

    # Compute score of the sequence from averaging predicted stage over a window centred on 0.5 for biomarker values
    # fig, ax = plotting.staged_biomarker_plots(dataloader=val_loader, net=net, device=DEVICE)
    # fig.show()
    window_seq_val = stages_to_sequence.stages_to_sequence_window(dataloader=val_loader, net=net).cpu()
    fig, ax = plotting.plot_predicted_sequence(gt_ordering=seq_gt, pred_ordering=window_seq_val)
    print("Stage averaged over window around 0.5 sequence score:", evaluate_sequence(window_seq_val, seq_gt))

    # -------------- TESTING -------------------
    biomarker_idx_samples = []
    stage_samples = []
    weights = []
    delta = 0.1
    # maps biomarker index to ground truth position in the progression sequence
    biomarker_true_positions = np.argsort(seq_gt)
    # # ... then flip it for visual consistency with the other positional variance diagrams
    # biomarker_true_positions = num_biomarkers - 1 - biomarker_true_positions
    with torch.no_grad():
        for X, _ in val_loader:
            preds = net.predict_stage(X)
            preds = preds.squeeze()

            # Get the indices of biomarker samples
            biomarker_counts = torch.sum(torch.abs(X - 0.5) <= delta, dim=0)
            biomarker_idx = torch.concat([torch.ones(biomarker_counts[i]) * biomarker_true_positions[i] for i in range(biomarker_counts.shape[0])])

            obs_idx = torch.where(torch.abs(X - 0.5) <= delta)  # Sets of X indices where X ~= 0.5
            preds_to_consider = torch.zeros(X.shape, device=DEVICE)  # elem (j, i) stores prediction j iff X[j, i] ~= 0.5
            preds_to_consider[obs_idx] = preds[obs_idx[0]]
            preds_to_consider = preds_to_consider.T[preds_to_consider.T > 0]  # the indices returned by np.where are in row major order, so transpose

            biomarker_idx_samples.append(biomarker_idx)
            stage_samples.append(preds_to_consider)

            biomarker_measurements = X[torch.abs(X - 0.5) <= delta]
            # weights are according to absolute distance from 0.5. linear for now. can think about fancy functions later - in which case maybe we don't use a delta and use a function that decays quickly to 0?
            decay_coeff = 5000  # coefficient describing intensity of exponential decay as measurement leaves 0.5
            w = torch.exp(-decay_coeff * (biomarker_measurements - 0.5) ** 2)
            weights.append(w)

            # print(biomarker_measurements)
            # print(w)
            # print()

            # for i in range(biomarker_samples[-1].shape[0]):
            #     print(biomarker_samples[-1][i], stage_samples[-1][i])
            #
            # print(X)

    biomarker_idx_samples = torch.concat(biomarker_idx_samples).cpu().numpy()
    stage_samples = torch.concat(stage_samples).cpu().numpy()
    weights = torch.concat(weights).cpu().numpy()

    # weights = None

    fig, ax = plotting.event_time_uncertainty_mat(seq_gt, stage_samples, biomarker_idx_samples, weights=weights)

    plt.show()
