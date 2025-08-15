import matplotlib.pyplot as plt
import torch
import numpy as np

from models import ae_stager, vae_stager


def staged_biomarker_plots(dataloader, net, device):
    # plots only the first 15 biomarkers.

    X = []
    preds = []

    # Get number of biomarkers
    n_biomarkers = next(iter(dataloader))[0].shape[1]
    num_biomarkers_to_plot = min(15, n_biomarkers)

    # Get data and corresponding predictions
    with torch.no_grad():
        for data, label in dataloader:
            X.append(data.cpu())
            preds.append(net.predict_stage(data))

    X = torch.concatenate(X).squeeze()
    preds = torch.concatenate(preds).squeeze().cpu()

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


def mean_data_against_discretised_stage(dataloader, net, device):
    # plots only the first few biomarkers.

    # Get number of biomarkers
    example, _ = next(iter(dataloader))
    n_biomarkers = example.shape[1]
    num_biomarkers_to_plot = min(10, n_biomarkers)

    X_per_stage = [[] for _ in range(n_biomarkers + 1)]

    # Get data and corresponding predictions
    with torch.no_grad():
        for X, _ in dataloader:
            preds = net.predict_stage(X)

            # Translate and group preds into 0, 1, ..., n_biomarkers
            preds_int = torch.round(preds * n_biomarkers).int()

            # Sort the observation data by predicted stage.
            for i in range(X.shape[0]):
                X_per_stage[preds_int[i]].append(X[i].cpu())

    mean_X_per_stage = np.zeros((n_biomarkers + 1, n_biomarkers))
    for stage in range(n_biomarkers + 1):
        # Skip if no data has been recorded with this predicted stage
        if len(X_per_stage[stage]) == 0:
            continue
        # print(np.array(X_per_stage[stage]).shape)
        mean_X_per_stage[stage] = np.array(X_per_stage[stage]).mean(axis=0)
    mean_X_per_stage = np.array(mean_X_per_stage)

    # Plot mean biomarker against predicted discrete stage on one figure
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle("Mean biomarker level against predicted discretised stage")

    for i in range(num_biomarkers_to_plot):
        ax.plot(np.arange(n_biomarkers + 1), mean_X_per_stage[:, i], label=f"Biomarker {i}")

    ax.set_xlabel("Stage")
    ax.set_ylabel("Biomarker level")
    ax.legend(loc="upper right")

    return fig, ax


def gt_biomarker_plots(dataloader):
    # plots only the first 15 biomarkers. biomarker level against gt pseudotime.

    X = []
    labels = []

    # Get number of biomarkers
    example, _ = next(iter(dataloader))
    num_biomarkers_to_plot = min(15, example.shape[1])

    # Get data and corresponding predictions
    with torch.no_grad():
        for data, label in dataloader:
            X.append(data.cpu())
            labels.append(label.cpu() / example.shape[1])

    X = torch.concatenate(X).squeeze()
    labels = torch.concatenate(labels).squeeze()

    # Plot each biomarker on separate axes of the same figure
    n_x = np.round(np.sqrt(num_biomarkers_to_plot)).astype(int)
    n_y = np.ceil(np.sqrt(num_biomarkers_to_plot)).astype(int)
    fig, ax = plt.subplots(n_y, n_x, figsize=(9, 9))
    fig.suptitle("Biomarker measurement against ground truth stage")

    for i in range(num_biomarkers_to_plot):
        ax.flat[i].scatter(labels, X[:, i])
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
            labels.append(label.cpu())
            preds.append(net.predict_stage(data))

    labels = torch.concatenate(labels).squeeze().cpu() / num_biomarkers
    preds = torch.concatenate(preds).squeeze().cpu()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(labels, preds)

    # Plot a straight line for comparison
    ax.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
    # ax.plot(preds - labels)

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


def plot_predicted_sequence(gt_ordering, pred_ordering):
    # Adapted from kde_ebm.plotting.mcmc_uncert_mat
    n_biomarkers = gt_ordering.shape[0]
    score_names = ['BM{}'.format(x) for x in range(n_biomarkers)]

    matrix = np.zeros((n_biomarkers, n_biomarkers))
    matrix[gt_ordering, pred_ordering] = 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(matrix, interpolation='nearest', cmap='Purples')

    if n_biomarkers > 50:
        stp = 5
    elif n_biomarkers > 20:
        stp = 2
    else:
        stp = 1

    tick_marks_x = np.arange(0, n_biomarkers, stp)
    x_labels = range(1, n_biomarkers + 1, stp)
    ax.set_xticks(tick_marks_x)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=8)

    if n_biomarkers <= 50:
        ax.set_yticks(np.arange(n_biomarkers))
        ax.set_yticklabels(np.array(score_names, dtype='object')[gt_ordering],
                           rotation=30, ha='right',
                           rotation_mode='anchor',
                           fontsize=8,
                           )
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    ax.set_ylabel("Biomarker in ground truth ordering")
    ax.set_xlabel("Predicted sequence position of event")
    fig.suptitle("Ground truth sequence position vs predicted position")

    return fig, ax


def event_time_uncertainty_mat(gt_ordering, x, y, weights=None):
    # Experimental: plot inferred event time distribution for each biomarker, as an uncertainty matrix.
    # Designed to look visually consistent with the output of kde_ebm.plotting.mcmc_uncert_mat
    # Arguments are np arrays
    # y here is the position of the biomarker in the ground truth sequence, just so the diagram is intuitive.

    n_biomarkers = gt_ordering.shape[0]
    score_names = ['BM{}'.format(x) for x in range(n_biomarkers)]

    fig, ax = plt.subplots(figsize=(8, 8))
    hist, xedges, yedges = np.histogram2d(x, y, bins=(n_biomarkers, n_biomarkers), weights=weights)
    matrix = hist.T
    matrix = matrix / np.sum(matrix, axis=1)[:, None]
    ax.imshow(matrix, interpolation='nearest', cmap='Purples')

    # Set y (biomarker) ticks and labels
    if n_biomarkers <= 50:
        ax.set_yticks(np.arange(n_biomarkers))
        ax.set_yticklabels(np.array(score_names, dtype='object')[gt_ordering],
                           rotation=30, ha='right',
                           rotation_mode='anchor',
                           fontsize=8,
                           )
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    # Set x (pseudotime) ticks and labels
    tick_marks_x = np.arange(0, n_biomarkers + 1, n_biomarkers / 10) - 0.5  # default 10 ticks on x axis
    ax.set_xticks(tick_marks_x)
    x_labels = [f'{x_label:.2f}' for x_label in np.linspace(np.min(x), np.max(x), tick_marks_x.shape[0])]

    ax.set_xticks(tick_marks_x)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=8)

    ax.set_ylabel("Biomarker (in order of ground truth sequence)")
    ax.set_xlabel("Predicted pseudotime of event")
    fig.suptitle("Event time uncertainty matrix")

    return fig, ax
