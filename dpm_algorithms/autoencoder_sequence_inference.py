import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from dpm_algorithms import plotting
from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_TEST_DIR, \
    SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_TEST_DIR
from models import ae_stager, vae_stager
from datasets.biomarker_dataset import BiomarkerDataset
from dpm_algorithms.evaluation import evaluate_autoencoder, evaluate_sequence


def infer_seq_from_network(dataloader, net):
    n_biomarkers = next(iter(dataloader))[0].shape[1]

    with torch.no_grad():
        # Estimate midpoints using the data and assigned pseudo-labels.
        # First get the minimum and maximum predicted stages, then divide that range into bins.
        min_prediction = float('inf')
        max_prediction = -float('inf')

        for X, _ in dataloader:
            preds = net.predict_stage(X)

            min_prediction = min(min_prediction, preds.min())
            max_prediction = max(max_prediction, preds.max())

        num_bins = 10
        X_per_bin = [[] for _ in range(num_bins)]
        for X, _ in dataloader:
            preds = net.predict_stage(X)

            # Scale preds into their fractional position from min_prediction to max_prediction.
            preds = (preds - min_prediction) / (max_prediction - min_prediction)

            # Translate and group preds into the desired number of windows
            preds_int = torch.round(preds * (num_bins - 1)).int()

            # Sort the observation data by predicted pseudo-stage.
            for i in range(preds_int.shape[0]):
                X_per_bin[preds_int[i]].append(X[i])

        # Get the minimum and maximum window averages found.
        mean_X_per_bin = []
        # for bin in range(num_bins):
        #     # Skip if no data has been recorded in this window
        #     if len(X_per_bin[bin]) == 0:
        #         X_per_bin[bin] = [torch.zeros(n_biomarkers, device=DEVICE)]
        #     mean_X_per_bin.append(torch.row_stack(X_per_bin[bin]).mean(dim=0))
        for bin in range(num_bins):
            # Use the previous bin's value if no data has been recorded with this predicted stage. If first stage, use 0
            if len(X_per_bin[bin]) == 0:
                if bin > 0:
                    X_per_bin[bin] = [mean_X_per_bin[bin - 1]]
                else:
                    X_per_bin[bin] = [torch.zeros(n_biomarkers, device=DEVICE)]
            mean_X_per_bin.append(torch.row_stack(X_per_bin[bin]).mean(dim=0))
        mean_X_per_bin = torch.row_stack(mean_X_per_bin)

        midpoints = 0.5

        # estimated_onset_times = torch.zeros(n_biomarkers, device=DEVICE)
        # counts = torch.zeros(n_biomarkers,
        #                      device=DEVICE)  # number of measurements per biomarker that have fallen in a window of 0.5 so far
        # total_weights = torch.zeros(n_biomarkers, device=DEVICE)
        # for X, _ in dataloader:
        #     preds = net.predict_stage(X)

            # # Get the predicted stage of Xs that are a percentage of the full range away from the corresponding mean
            # # with delta being that percentage
            # preds = preds.squeeze(-1)
            # prev_counts = counts.detach().clone()
            # counts += torch.sum(torch.abs(X - midpoints) <= delta * (AD_estimation - CN_estimation), dim=0)
            # obs_idx = torch.where(
            #     torch.abs(X - midpoints) <= delta * (AD_estimation - CN_estimation))  # Sets of X indices where X ~= 0.5
            # preds_to_consider = torch.zeros(X.shape,
            #                                 device=DEVICE)  # elem (j, i) stores prediction j iff X[j, i] ~= 0.5
            # preds_to_consider[obs_idx] = preds[obs_idx[0]]
            # idx_to_adjust = torch.unique(obs_idx[1])  # indices of biomarkers whose counts need adjusting
            # estimated_onset_times[idx_to_adjust] = (estimated_onset_times[idx_to_adjust] * prev_counts[idx_to_adjust] +
            #                                    torch.sum(preds_to_consider, dim=0)[idx_to_adjust]) / counts[
            #                                       idx_to_adjust]

            # # Compute a weighted sum instead, with weights being a function of prediction and midpoint.
            # decay_coeff = 1000  # describes intensity of exponential decay as measurement leaves the midpoint
            # w = torch.exp(-decay_coeff * (X - midpoints) ** 2)
            # weighted_preds = (preds * w).sum(0)
            # total_weights += w.sum(0)
            # estimated_onset_times += weighted_preds

    # estimated_onset_times /= total_weights

    # Encoder ver
    # Search over the windows. Interpolate between the bins of the closest values on either side of each midpoint
    temp = mean_X_per_bin - midpoints
    temp[temp < 0] = float('inf')
    estimated_stage_above = torch.argmin(temp, dim=0)
    estimated_midpoint_above = mean_X_per_bin[estimated_stage_above, torch.arange(n_biomarkers)]
    temp = midpoints - mean_X_per_bin
    temp[temp < 0] = float('inf')
    estimated_stage_below = torch.argmin(temp, dim=0)
    estimated_midpoint_below = mean_X_per_bin[estimated_stage_below, torch.arange(n_biomarkers)]

    # Scale estimations back onto pseudotime latent
    estimated_stage_above = min_prediction + (max_prediction - min_prediction) * (estimated_stage_above / (num_bins - 1))
    estimated_stage_below = min_prediction + (max_prediction - min_prediction) * (estimated_stage_below / (num_bins - 1))

    # estimated_stage_above = torch.argmin(torch.max(torch.zeros_like(mean_X_per_bin), mean_X_per_bin - midpoints), dim=0)
    # estimated_stage_below = torch.argmin(torch.max(torch.zeros_like(mean_X_per_bin), midpoints - mean_X_per_bin), dim=0)

    estimated_onset_times = estimated_stage_below + ((estimated_stage_above - estimated_stage_below) *
                                                     ((0.5 - estimated_midpoint_below) / (
                                                             estimated_midpoint_above - estimated_midpoint_below)))
    # estimated_onset_times = torch.argmin(torch.abs(mean_X_per_bin - midpoints), dim=0).double()
    # estimated_onset_times = min_prediction + (max_prediction - min_prediction) * (estimated_onset_times / num_bins)  # scale back onto pseudotime

    # for i in range(num_bins):
    #     print(mean_X_per_bin[i])

    # print(estimated_onset_times)

    # Outputs one figure of biomarker trajectories computed by averaging biomarkers over binned encoder outputs.
    # plots only the first few biomarkers.
    # normalised: if True, scales every biomarker trajectory between 0 and 1 inclusive.

    # # Plot mean biomarker against predicted discrete stage on one figure
    # fig, ax = plt.subplots(figsize=(12, 9))
    # fig.suptitle("Mean biomarker level against predicted discretised stage")
    #
    # colors = ['C{}'.format(x) for x in range(10)]
    # for i in range(n_biomarkers):
    #     if i < 10:
    #         linestyle = "solid"
    #     else:
    #         linestyle = "dotted"
    #
    #     ax.plot(np.linspace(min_prediction.cpu(), max_prediction.cpu(), num_bins),
    #             mean_X_per_bin[:, i].cpu(), label=dataloader.dataset.biomarker_names[i], color=colors[i % 10],
    #             linestyle=linestyle)
    #     # ax.scatter(np.linspace(min_prediction.cpu(), max_prediction.cpu(), num_bins),
    #     #            mean_X_per_bin[:, i].cpu(), c=colors[i % 10])
    #     ax.scatter(estimated_onset_times[i].cpu(), 0.5 * np.ones(1), c=colors[i % 10])
    #
    #     # ax.scatter(np.array([estimated_stage_below[i].cpu(), estimated_stage_above[i].cpu()]),
    #     #            np.array([estimated_midpoint_below[i].cpu(), estimated_midpoint_above[i].cpu()]), c=colors[i % 10])
    #
    # ax.set_xlabel("Stage")
    # ax.set_ylabel("Biomarker level")
    # ax.legend(loc="upper right")
    # fig.show()
    #
    # print(estimated_onset_times)

    return torch.argsort(estimated_onset_times)


if __name__ == "__main__":
    # USER CONFIGURATION --------------------
    dataset_name = "rexample_0"
    model_name = dataset_name

    model_type = "ae"  # only vae or ae supported currently
    if model_type not in ["vae", "ae"]:
        print("Model type must be one of 'vae' or 'ae' (case-sensitive)")
        exit()
    # ---------------------------------------

    # Load datasets
    train_dataset = BiomarkerDataset(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TRAIN_DIR,
                                     label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = BiomarkerDataset(dataset_names=dataset_name, obs_directory=SIMULATED_OBS_TEST_DIR,
                                   label_directory=SIMULATED_LABEL_TEST_DIR)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

    example_x, _ = next(iter(train_loader))
    num_biomarkers = example_x.shape[1]

    # Read in ground truth sequence - note that the val and train sets share one sequence
    seq_label_file = os.path.join(SIMULATED_LABEL_TRAIN_DIR, dataset_name + "_seq.json")
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

    window_seq_val = infer_seq_from_network(dataloader=val_loader, net=net).cpu()
    fig, ax = plotting.plot_predicted_sequence(gt_ordering=seq_gt, pred_ordering=window_seq_val)
    print("Kendall's tau :", evaluate_sequence(window_seq_val, seq_gt))

    plt.show()
