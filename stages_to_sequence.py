import torch
import numpy as np
from tqdm import tqdm

import plotting
from config import DEVICE


def stages_to_sequence_direct(num_biomarkers, dataloader, net, device):
    # Infer underlying sequence from latent stage predictions.
    # By this formula, a later score implies earlier abnormality and thus earlier position.
    # But it still depends on the direction of the learned latent...
    biomarker_scores = torch.zeros(num_biomarkers, device=device, requires_grad=False)

    # First do a pass through the data to estimate normal and abnormal levels of the biomarkers
    start = torch.zeros(num_biomarkers, device=device, requires_grad=False)  # estimate of biomarker levels at the start
    end = torch.zeros(num_biomarkers, device=device, requires_grad=False)  # estimate of biomarker levels at the end

    start_count = 0
    end_count = 0

    for X, _ in dataloader:
        X = X.to(device)
        pred = net.predict_stage(X)

        # consider measurements predicted to be within stages 0 to 3 and num_biomarkers - 3 to num_biomarkers?
        idx_within_start = torch.where(pred < 5 / num_biomarkers)[0]  # extract single tensor from tuple
        idx_within_end = torch.where(pred > 1 - (5 / num_biomarkers))[0]

        start_count += idx_within_start.shape[0]
        end_count += idx_within_end.shape[0]

        if start_count > 0:
            start = start * (start_count - idx_within_start.shape[0]) / start_count \
                + (X[idx_within_start] * (1 - pred[idx_within_start])).sum(0) / start_count
        if end_count > 0:
            end = end * (end_count - idx_within_end.shape[0]) / end_count \
                + (X[idx_within_end] * pred[idx_within_end]).sum(0) / end_count

    reverse = torch.ones(num_biomarkers).to(device)
    reverse[torch.where(start > end)] = -1  # indicating that the level decreases with abnormality

    midpoints = start + (end - start) / 2

    corrected_start = torch.min(start, end)
    corrected_end = torch.max(start, end)

    # print(corrected_start)
    # print(corrected_end)

    for X, labels in dataloader:
        X = X.to(device)
        pred = net.predict_stage(X)

        # Scale X using prediction and start/end.
        # first flip around the midpoints so start < end
        X = midpoints + reverse * X - midpoints

        # then clip
        X = torch.max(X, corrected_start)
        X = torch.min(X, corrected_end)

        # rescale
        X = (X - corrected_start) / (corrected_end - corrected_start)

        biomarker_scores += (pred * X).sum(dim=0)

    return torch.argsort(biomarker_scores, descending=True)


def sigmoid(x, a, b, c, d):
    return a / (1 + torch.exp(-50 * b * (x - c))) + d


def stages_to_sequence_window(dataloader, net):
    n_biomarkers = next(iter(dataloader))[0].shape[1]
    # onset_times = [[] for _ in range(n_biomarkers)]

    counts = torch.zeros(n_biomarkers, device=DEVICE)  # number of measurements per biomarker that have fallen in a window of 0.5 so far
    mean_onset_times = torch.zeros(n_biomarkers, device=DEVICE)
    delta = 0.1

    with torch.no_grad():
        for X, _ in dataloader:
            preds = net.predict_stage(X)

            # Get the predicted stage of observation data within a window of 0.5 (i.e. +- delta)
            # for i in range(X.shape[1]):
            #     if (torch.abs(X[:, i] - 0.5) <= delta).sum() > 0:
            #         onset_times[i].append(preds[torch.abs(X[:, i] - 0.5) <= delta])

            # # try 2
            # for i in range(X.shape[1]):
            #     onset_times[i].append(preds[torch.abs(X[:, i] - 0.5) <= delta])

            # # try 3
            preds = preds.squeeze()
            prev_counts = counts.detach().clone()
            counts += torch.sum(torch.abs(X - 0.5) <= delta, dim=0)
            obs_idx = torch.where(torch.abs(X - 0.5) <= delta)  # Sets of X indices where X ~= 0.5
            preds_to_consider = torch.zeros(X.shape, device=DEVICE)  # elem (j, i) stores prediction j iff X[j, i] ~= 0.5
            preds_to_consider[obs_idx] = preds[obs_idx[0]]
            idx_to_adjust = torch.unique(obs_idx[1])  # indices of biomarkers whose counts need adjusting
            mean_onset_times[idx_to_adjust] = (mean_onset_times[idx_to_adjust] * prev_counts[idx_to_adjust] +
                                               torch.sum(preds_to_consider, dim=0)[idx_to_adjust]) / counts[idx_to_adjust]

    # # Average the onset times for each biomarker
    # mean_onset_times = torch.zeros(n_biomarkers)
    # for i in range(n_biomarkers):
    #     mean_onset_times[i] = torch.concatenate(onset_times[i]).mean()

    return torch.argsort(mean_onset_times)


def fit_biomarker_curves(dataloader, net, n_epochs, device, lr=0.01):
    # Fit sigmoids to each biomarker using SGD

    # Get number of biomarkers from dataset
    example_x, _ = next(iter(dataloader))
    num_biomarkers = example_x.shape[1]

    # Initialise hyperparameters and optimiser
    sigmoid_params = torch.nn.Parameter(torch.tensor([1., 0., 0.5, 0.], device=device).repeat((num_biomarkers, 1)), requires_grad=True)
    opt = torch.optim.SGD([sigmoid_params], lr=lr)

    dataset_size = len(dataloader.dataset)
    prev_loss = float('inf')
    patience = 5  # number of epochs to wait without improvement
    epochs_without_improvement = 0

    for i in tqdm(range(n_epochs), desc=f"Fitting biomarker curves"):
        overall_loss = 0
        for data, _ in dataloader:
            data = data.to(device)
            pred = net.predict_stage(data)

            opt.zero_grad()

            loss = ((data - sigmoid(pred, sigmoid_params[:, 0], sigmoid_params[:, 1], sigmoid_params[:, 2], sigmoid_params[:, 3])) ** 2).mean()
            overall_loss += loss * data.shape[0] / dataset_size

            loss.backward()
            opt.step()

        # Track the number of epochs without improvement and quit if necessary
        if overall_loss >= prev_loss:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Stopping curve-fitting early due to lack of observed improvement")
                break
        else:
            epochs_without_improvement = 0

            # Also exit early if loss improvement falls below a threshold
            if prev_loss - overall_loss < 1e-4:
                print("Stopping curve-fitting early due to small improvements")
                break

        prev_loss = overall_loss

        if i % 10 == 0:
            print(f"epoch {i}: loss {overall_loss:.4f}")

    # print(sigmoid_params.grad)

    # plot fitted curves on top of scatter plots
    fig, ax = plotting.staged_biomarker_plots(dataloader, net, device)
    t = np.linspace(0, 1, 1000)
    for i in range(min(num_biomarkers, len(ax.flat))):
        y = sigmoid(torch.tensor(t, device=device), sigmoid_params[i][0], sigmoid_params[i][1], sigmoid_params[i][2], sigmoid_params[i][3])
        y = y.cpu().detach().numpy()
        ax.flat[i].plot(t, y, color="red")

    return sigmoid_params, fig, ax


def infer_seq_from_biomarker_curves(sigmoid_parameters):
    # Return, as the predicted sequence, the ascending order of biomarkers wrt the x offset of each fitted curve
    return torch.argsort(sigmoid_parameters[:, 2])

