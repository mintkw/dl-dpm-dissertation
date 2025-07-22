import torch
import scipy
import numpy as np


def compute_reconstruction_mse(dataloader, net, device):
    reconstruction_errors = []
    with torch.no_grad():
        for X, label in dataloader:
            X = X.to(device)
            reconstruction = net.reconstruct(X)
            reconstruction_errors.append((reconstruction - X) ** 2)

    reconstruction_errors = torch.concatenate(reconstruction_errors).to(device)
    mean_reconstruction_error = torch.mean(reconstruction_errors)

    return mean_reconstruction_error


def evaluate_autoencoder(dataloader, net, device):
    """

    Args:
        dataloader:
        net: needs an autoencoder
        device:
        num_latents: if > 1, evaluates the first dimension

    Returns:

    """
    predictions = []
    gt_stages = []
    reconstruction_errors = []
    n_biomarkers = next(iter(dataloader))[0].shape[1]
    with torch.no_grad():
        for X, label in dataloader:
            gt_stages.append(label)

            X = X.to(device)
            raw_preds = net.encode(X)
            predictions.append(raw_preds)
            reconstruction = net.reconstruct(X)
            reconstruction_errors.append((reconstruction - X) ** 2)

    predictions = torch.concatenate(predictions).squeeze().to(device)
    # scale predictions with number of biomarkers
    predictions *= n_biomarkers

    gt_stages = torch.concatenate(gt_stages).to(device)
    # gt_stages /= torch.max(gt_stages)

    rmse_stage_error = torch.sqrt(torch.mean((predictions - gt_stages) ** 2))

    reconstruction_errors = torch.concatenate(reconstruction_errors).to(device)
    reconstruction_error = torch.mean(reconstruction_errors)

    return rmse_stage_error, reconstruction_error


def evaluate_sequence(preds, gt):
    # Returns the Kendall's tau distance between the predicted and ground truth sequences. Expects both as a flat numpy array.
    return scipy.stats.kendalltau(np.argsort(preds), np.argsort(gt)).statistic
