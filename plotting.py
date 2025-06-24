import matplotlib.pyplot as plt
import torch
import os
import numpy as np

from config import DEVICE, PLOT_DIR, MODEL_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_VAL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_VAL_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
import ae_stager


def staged_biomarker_plots(dataloader, num_biomarkers, net, device, fit_curves=False):
    X = []
    preds = []
    # labels = []

    # Get data and corresponding predictions
    with torch.no_grad():
        for data, label in dataloader:
            X.append(data)
            preds.append(net.encode(data.to(device)))
            # labels.append(label)

    X = torch.concatenate(X).squeeze()
    preds = torch.concatenate(preds).squeeze().cpu()
    # labels = torch.concatenate(labels).squeeze()

    # # otherwise, plot all separately
    # for biomarker in range(num_biomarkers):
    #     plt.scatter(preds, X[:, biomarker])
    #     plt.xlabel("Predicted stage")
    #     plt.ylabel("Biomarker measurement")
    #     plt.title(f"Biomarker {biomarker}")
    #     plt.show()

    # Plot each biomarker on separate axes of the same figure
    n_x = np.round(np.sqrt(num_biomarkers)).astype(int)
    n_y = np.ceil(np.sqrt(num_biomarkers)).astype(int)
    fig, ax = plt.subplots(n_y, n_x, figsize=(9, 9))
    fig.suptitle("Biomarker measurement against predicted stage")

    for i in range(num_biomarkers):
        ax.flat[i].scatter(preds, X[:, i])
        ax.flat[i].set_title(f"Biomarker {i}")

    # Delete unused axes
    for j in range(num_biomarkers, n_x*n_y):
        fig.delaxes(ax.flat[j])
    fig.tight_layout()

    return fig, ax


def predicted_stage_comparison(dataloader, num_biomarkers, net, device):
    # Plot predicted stage against (normalised) ground truth.

    labels = []
    preds = []
    with torch.no_grad():
        for data, label in dataloader:
            labels.append(label)
            preds.append(net.encode(data.to(device)))

    labels = torch.concatenate(labels).squeeze().cpu() / num_biomarkers
    preds = torch.concatenate(preds).squeeze().cpu()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(labels, preds)
    ax.set_title("Predicted vs ground truth stages")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")

    return fig, ax
