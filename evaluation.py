import torch
import scipy
import numpy as np


def evaluate_autoencoder(dataloader, net, device):
    """

    Args:
        dataloader:
        net: needs an autoencoder
        device:

    Returns:

    """
    predictions = []
    gt_stages = []
    reconstruction_errors = []
    with torch.no_grad():
        for X, label in dataloader:
            gt_stages.append(label)

            X = X.to(device)
            raw_preds = net.encode(X)
            predictions.append(raw_preds)

            reconstruction = net.reconstruct(X)

            reconstruction_errors.append((reconstruction - X) ** 2)

    # scale predictions
    predictions = torch.concatenate(predictions).squeeze().to(device)
    # predictions = (predictions - torch.min(predictions)) / (torch.max(predictions) - torch.min(predictions))
    # predictions = torch.sigmoid(predictions)

    # scale stages
    gt_stages = torch.concatenate(gt_stages).to(device)
    gt_stages /= torch.max(gt_stages)

    reconstruction_errors = torch.concatenate(reconstruction_errors).to(device)

    mse_stage_error = torch.min(torch.mean((predictions - gt_stages) ** 2))

    reconstruction_error = torch.mean(reconstruction_errors)

    return mse_stage_error, reconstruction_error


def evaluate_sequence(preds, gt):
    # Returns the Kendall's tau distance between the predicted and ground truth sequences. Expects both as a flat numpy array.
    return scipy.stats.kendalltau(np.argsort(preds), np.argsort(gt)).statistic
