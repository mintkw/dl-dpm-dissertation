import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import plotting
from config import DEVICE, MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_OBS_VAL_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_LABEL_VAL_DIR, PLOT_DIR
import vae_stager
import ae_stager
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from evaluation import evaluate_autoencoder, evaluate_sequence


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
        pred = net.encode(X)

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
        pred = net.encode(X)

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


def sigmoid(x, a, b, c):
    return a / (1 + torch.exp(-b * (x - c)))


def inverse_sigmoid(y, a, b, c):
    return -torch.log((a - y) / y) / b + c


def fit_biomarker_curves(dataloader, net, n_epochs, device, lr=0.01):
    # Fit sigmoids to each biomarker using SGD

    # initialise hyperparameters and optimiser
    sigmoid_params = torch.nn.Parameter(torch.tensor([1., 10., 0.5], device=device).repeat((num_biomarkers, 1)), requires_grad=True)
    opt = torch.optim.SGD([sigmoid_params], lr=lr)

    dataset_size = len(dataloader.dataset)
    prev_loss = float('inf')
    patience = 5  # number of epochs to wait without improvement
    epochs_without_improvement = 0

    for i in range(n_epochs):
        overall_loss = 0
        for data, _ in dataloader:
            data = data.to(device)
            pred = net.encode(data)

            loss = ((data - sigmoid(pred, sigmoid_params[:, 0], sigmoid_params[:, 1], sigmoid_params[:, 2])) ** 2).mean()
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
    for i in range(num_biomarkers):
        y = sigmoid(torch.tensor(t, device=device), sigmoid_params[i][0], sigmoid_params[i][1], sigmoid_params[i][2])
        y = y.cpu().detach().numpy()
        ax.flat[i].plot(t, y)

    return sigmoid_params, fig, ax


def infer_seq_from_biomarker_curves(sigmoid_parameters):
    # Compute the points at which each biomarker is estimated to cross 0.5
    midpoints = inverse_sigmoid(0.5, sigmoid_parameters[:, 1], sigmoid_parameters[:, 2], sigmoid_parameters[:, 2])

    # Return, as the predicted sequence, the order in which the biomarkers cross 0.5
    return torch.argsort(midpoints)


if __name__ == "__main__":
    dataset_name = "synthetic_120_10_dpm_0"

    # Load datasets
    train_dataset = SyntheticDatasetVec(dataset_name=dataset_name, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataset = SyntheticDatasetVec(dataset_name=dataset_name, obs_directory=SIMULATED_OBS_VAL_DIR, label_directory=SIMULATED_LABEL_VAL_DIR)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

    example_x, _ = next(iter(train_loader))
    num_biomarkers = example_x.shape[1]

    # Read in ground truth sequence - note that the val and train sets share one sequence
    seq_label_file = os.path.join(SIMULATED_LABEL_TRAIN_DIR, dataset_name + "_seq.json")
    with open(seq_label_file, 'r') as f:
        seq_gt = json.load(f)

    # flatten sequence
    seq_gt = np.array(seq_gt).squeeze(1)

    print("gt sequence:\n", seq_gt)

    # # -------- VAE ----------
    # vae_enc = vae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    # vae_dec = vae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)
    #
    # vae = vae_stager.VAE(enc=vae_enc, dec=vae_dec)
    #
    # # Load a model fitted to the particular dataset
    # enc_model_path = os.path.join(MODEL_DIR, "vae", "enc_" + dataset_name + ".pth")
    # dec_model_path = os.path.join(MODEL_DIR, "vae", "dec_" + dataset_name + ".pth")
    #
    # vae_enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
    # vae_dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))
    #
    # # Verify that everything loaded alright by printing mean squared error on training set
    # print("Mean squared error on training set:", evaluate_autoencoder(train_loader, vae, DEVICE)[0])
    #
    # # Infer sequence from stage information
    # vae_seq_prediction = stages_to_sequence_direct(num_biomarkers, train_loader, vae, DEVICE)
    #
    # # Unfortunately I can't constrain the order so compute the score of both reversed and forward orders
    # # and return the higher-scoring order
    # vae_seq_predictions = [vae_seq_prediction, torch.flip(vae_seq_prediction, dims=(0,))]
    # higher_score = torch.argmax(torch.tensor([evaluate_sequence(vae_seq_predictions[0].cpu(), seq_gt),
    #                                           evaluate_sequence(vae_seq_predictions[1].cpu(), seq_gt)])).item()
    #
    # print("seq inferred from VAE predictions:")
    # print(vae_seq_predictions[higher_score])
    # print("VAE sequence score:", evaluate_sequence(vae_seq_predictions[higher_score].cpu(), seq_gt))

    # --------- AE ----------
    ae_enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    ae_dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

    ae = ae_stager.AE(enc=ae_enc, dec=ae_dec)

    # Load a model fitted to the particular dataset
    enc_model_path = os.path.join(MODEL_DIR, "ae", "enc_" + dataset_name + ".pth")
    dec_model_path = os.path.join(MODEL_DIR, "ae", "dec_" + dataset_name + ".pth")

    ae_enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
    ae_dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

    # Verify that everything loaded alright by printing mean squared error on training set
    print("Mean squared error on training set:", evaluate_autoencoder(train_loader, ae, DEVICE)[0])

    # 1. Infer sequence directly from stage information
    ae_seq_prediction_train = stages_to_sequence_direct(num_biomarkers, train_loader, ae, DEVICE)
    ae_seq_prediction_val = stages_to_sequence_direct(num_biomarkers, val_loader, ae, DEVICE)

    # Unfortunately I can't constrain the order so compute the score of both reversed and forward orders
    # and return the higher-scoring order
    ae_seq_predictions_train = [ae_seq_prediction_train, torch.flip(ae_seq_prediction_train, dims=(0,))]
    higher_score = torch.argmax(torch.tensor([evaluate_sequence(ae_seq_predictions_train[0].cpu(), seq_gt),
                                              evaluate_sequence(ae_seq_predictions_train[1].cpu(), seq_gt)])).item()

    print("seq inferred directly from AE predictions on training set:")
    print(ae_seq_predictions_train[higher_score])
    print("AE sequence score:", evaluate_sequence(ae_seq_predictions_train[higher_score].cpu(), seq_gt))

    ae_seq_predictions_val = [ae_seq_prediction_val, torch.flip(ae_seq_prediction_val, dims=(0,))]
    higher_score = torch.argmax(torch.tensor([evaluate_sequence(ae_seq_predictions_val[0].cpu(), seq_gt),
                                              evaluate_sequence(ae_seq_predictions_val[1].cpu(), seq_gt)])).item()

    print("seq inferred directly from AE predictions on validation set:")
    print(ae_seq_predictions_val[higher_score])
    print("AE sequence score:", evaluate_sequence(ae_seq_predictions_val[higher_score].cpu(), seq_gt))

    # 2. Infer sequence by fitting biomarker curves
    # Fit biomarker curves
    sigmoid_params_train, fig_train, ax_train = fit_biomarker_curves(train_loader, ae, n_epochs=100, device=DEVICE, lr=0.001)
    label = fig_train._suptitle.get_text()
    fig_train.suptitle(label + " on training set")
    fig_train.show()

    sigmoid_params_val, fig_val, ax_val = fit_biomarker_curves(val_loader, ae, n_epochs=100, device=DEVICE, lr=0.001)
    label = fig_val._suptitle.get_text()
    fig_val.suptitle(label + " on validation set")
    fig_val.show()

    # Infer sequence from the fitted curves
    curve_fitting_seq_train = infer_seq_from_biomarker_curves(sigmoid_params_train)
    curve_fitting_seq_val = infer_seq_from_biomarker_curves(sigmoid_params_val)
    print("seq inferred from fitted biomarker curves on training set:")
    print(curve_fitting_seq_train)
    print("Curve-fitted sequence score:", evaluate_sequence(curve_fitting_seq_train.cpu(), seq_gt))

    print("seq inferred from fitted biomarker curves on validation set:")
    print(curve_fitting_seq_val)
    print("Curve-fitted sequence score:", evaluate_sequence(curve_fitting_seq_val.cpu(), seq_gt))

    plt.show()
