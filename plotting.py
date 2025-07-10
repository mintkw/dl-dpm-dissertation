import matplotlib.pyplot as plt
import torch
import numpy as np

from models import ae_stager, vae_stager


def staged_biomarker_plots(dataloader, net, device):
    # plots only the first 10 biomarkers.

    X = []
    preds = []
    # labels = []

    # Get number of biomarkers
    example, _ = next(iter(dataloader))
    num_biomarkers_to_plot = min(15, example.shape[1])

    # Get data and corresponding predictions
    with torch.no_grad():
        for data, label in dataloader:
            X.append(data)
            preds.append(net.encode(data.to(device)))
            # labels.append(label)

    X = torch.concatenate(X).squeeze()
    preds = torch.concatenate(preds).squeeze().cpu()
    # labels = torch.concatenate(labels).squeeze()

    # Plot each biomarker on separate axes of the same figure
    n_x = np.round(np.sqrt(num_biomarkers_to_plot)).astype(int)
    n_y = np.ceil(np.sqrt(num_biomarkers_to_plot)).astype(int)
    fig, ax = plt.subplots(n_y, n_x, figsize=(9, 9))
    fig.suptitle("Biomarker measurement against predicted stage")

    for i in range(num_biomarkers_to_plot):
        ax.flat[i].scatter(preds, X[:, i])
        ax.flat[i].set_title(f"Biomarker {i}")

    # Delete unused axes
    for j in range(num_biomarkers_to_plot, n_x*n_y):
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
    fig.suptitle("Predicted vs ground truth stages")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")

    return fig, ax


def predicted_biomarker_trajectories(num_biomarkers, net, device):
    # net: must be a VAE or AE object.

    # Given an autoencoder, plot the (expected) biomarker level against pseudo-time as output by the decoder.
    # Plots the first 10 biomarkers.
    num_biomarkers_to_plot = min(15, num_biomarkers)

    t = torch.linspace(0, 1, 100).to(device)

    # automatically handle whether the net is a vae or ae.
    with torch.no_grad():
        if type(net) == vae_stager.VAE:
            output = net.dec(t.unsqueeze(1))
            preds = output.mean
            sigma = output.variance
        elif type(net) == ae_stager.AE:
            preds = net.dec(t.unsqueeze(1))
            sigma = torch.zeros(preds.shape)
        else:
            print("net should be a VAE or AE object")
            return

    t = t.cpu()
    preds = preds.cpu()
    sigma = sigma.cpu()

    # Plot each biomarker on separate axes of the same figure
    n_x = np.round(np.sqrt(num_biomarkers_to_plot)).astype(int)
    n_y = np.ceil(np.sqrt(num_biomarkers_to_plot)).astype(int)
    fig, ax = plt.subplots(n_y, n_x, figsize=(9, 9))
    fig.suptitle("Predicted biomarker trajectories")

    for i in range(num_biomarkers_to_plot):
        ax.flat[i].scatter(t, preds[:, i].cpu())
        ax.flat[i].set_title(f"Biomarker {i}")

        ax.flat[i].fill_between(
            t,
            preds[:, i] - np.sqrt(sigma[:, i]),
            preds[:, i] + np.sqrt(sigma[:, i]),
            alpha=0.5,
            label=r"standard deviation",
        )

    # Delete unused axes
    for j in range(num_biomarkers_to_plot, n_x * n_y):
        fig.delaxes(ax.flat[j])
    fig.tight_layout()

    return fig, ax

