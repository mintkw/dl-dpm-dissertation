import matplotlib.pyplot as plt
import torch

from config import SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_OBS_TEST_DIR, SIMULATED_LABEL_TEST_DIR, DEVICE
from datasets.synthetic_dataset import SyntheticDataset


def plot_dataset_in_latent_space(net, dataset_names):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("Encoding of the training set in the latent space")

    # Load the data from each subtype individually so they can be differentiated through colour coding
    train_datasets = [SyntheticDataset(dataset_names=[dataset_name],
                                       obs_directory=SIMULATED_OBS_TRAIN_DIR,
                                       label_directory=SIMULATED_LABEL_TRAIN_DIR) for dataset_name in dataset_names]
    train_loaders = [torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True) for dataset in train_datasets]
    cmaps = ['viridis', 'inferno', 'cool']

    for i in range(len(train_loaders)):
        loader = train_loaders[i]
        stage_latents = []
        subtype_latents = []
        labels = []
        for X, stage in loader:
            stage_latents.append(net.predict_stage(X))
            subtype_latents.append(net.predict_subtype(X))
            # latents.append(net.encode(X))
            labels.append(stage)
        # latents = torch.concatenate(latents, dim=0).detach().cpu()
        stage_latents = torch.concatenate(stage_latents, dim=0).detach().cpu()
        subtype_latents = torch.concatenate(subtype_latents, dim=0).detach().cpu()

        labels = torch.concatenate(labels, dim=0).detach().cpu()

        colourmap = ax.scatter(stage_latents, subtype_latents, c=labels, cmap=cmaps[i])
        fig.colorbar(colourmap, ax=ax)

    return fig, ax


def compute_subtype_accuracy(net, dataset_names):
    """
    Computes subtype accuracy (clustering accuracy) on a validation set.

    Args:
        net:
        dataset_names:

    Returns:

    """
    # Load the data from each subtype individually
    val_datasets = [SyntheticDataset(dataset_names=[dataset_name],
                                     obs_directory=SIMULATED_OBS_TEST_DIR,
                                     label_directory=SIMULATED_LABEL_TEST_DIR) for dataset_name in dataset_names]
    val_loaders = [torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True) for dataset in val_datasets]

    subtype_accuracy = 0.0
    with torch.no_grad():
        for i in range(len(dataset_names)):
            val_loader = val_loaders[i]
            for X, _ in val_loader:
                subtype_preds = net.predict_subtype(X)
                gt_preds = torch.tensor([i for _ in range(X.shape[0])], device=DEVICE)

                subtype_accuracy += torch.sum(subtype_preds == gt_preds)

    subtype_accuracy /= sum([len(val_dataset) for val_dataset in val_datasets])

    return subtype_accuracy


def compute_subtype_accuracy_with_cluster_mapping(net, dataset_names):
    """
    Computes subtype accuracy (clustering accuracy) on a validation set
    using the evaluation protocol of Makhzani et al. (2015).

    Args:
        net:
        dataset_names:

    Returns:

    """
    n_subtypes = len(dataset_names)

    # Load the data from each subtype individually
    val_datasets = [SyntheticDataset(dataset_names=[dataset_name],
                                     obs_directory=SIMULATED_OBS_TEST_DIR,
                                     label_directory=SIMULATED_LABEL_TEST_DIR) for dataset_name in dataset_names]
    val_loaders = [torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True) for dataset in val_datasets]

    max_probs = torch.zeros(n_subtypes, device=DEVICE)  # elem i holds max_n{q(y=i|x_n)}
    cluster_mapping = torch.arange(n_subtypes, device=DEVICE)  # elem i holds the mapping from model cluster index (y_i) to dataset subtype index

    subtype_accuracy = 0.0
    subtype_preds = []
    gt_preds = []
    with torch.no_grad():
        # Pass through the dataset to compute max_probs and cluster_mapping
        for i in range(len(dataset_names)):
            val_loader = val_loaders[i]

            for X, _ in val_loader:
                q_pi = net.subtype_scores(X)

                # Update max probabilities per class
                max_probs, indices = torch.max(torch.row_stack([max_probs, q_pi]), dim=0)

                # Update cluster mapping where indices != 0
                cluster_mapping[indices != 0] = i

                subtype_preds.append(net.predict_subtype(X))
                gt_preds.append(torch.tensor([i for _ in range(X.shape[0])], device=DEVICE))

        subtype_preds = torch.concatenate(subtype_preds, dim=-1)
        gt_preds = torch.concatenate(gt_preds, dim=-1)

    # print(max_probs, cluster_mapping)

    mapped_subtype_preds = subtype_preds
    for i in range(n_subtypes):
        mapped_subtype_preds[subtype_preds == i] = cluster_mapping[i]

    subtype_accuracy = torch.sum(mapped_subtype_preds == gt_preds) / torch.numel(mapped_subtype_preds)

    return subtype_accuracy
