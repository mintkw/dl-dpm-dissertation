import torch


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
    predictions = (predictions - torch.min(predictions)) / (torch.max(predictions) - torch.min(predictions))
    # predictions = torch.sigmoid(predictions)

    # scale stages
    gt_stages = torch.concatenate(gt_stages).to(device)
    gt_stages /= torch.max(gt_stages)

    reconstruction_errors = torch.concatenate(reconstruction_errors).to(device)

    # latent can scale in an opposite direction to stages, so try both directions and take the min error
    mse_stage_error = torch.min(torch.mean((predictions - gt_stages) ** 2),
                                torch.mean((predictions - 1 + gt_stages) ** 2))

    reconstruction_error = torch.mean(reconstruction_errors)

    return mse_stage_error, reconstruction_error
