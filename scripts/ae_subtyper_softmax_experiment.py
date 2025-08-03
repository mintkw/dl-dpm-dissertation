import os

import torch
import torch.nn as nn
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TRAIN_DIR, SIMULATED_OBS_VAL_DIR, SIMULATED_LABEL_VAL_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from plotting import predicted_stage_comparison
from models import ae_stager
import evaluation
from subtyping_utils import plot_dataset_in_latent_space, compute_subtype_accuracy, compute_subtype_accuracy_with_cluster_mapping


class Encoder(nn.Module):

    def __init__(self, n_biomarkers, d_latent, n_subtypes):
        super().__init__()
        self.d_latent = d_latent
        self.n_subtypes = n_subtypes
        self.n_biomarkers = n_biomarkers
        self.net = nn.Sequential(nn.Linear(self.n_biomarkers, 16),
                                 nn.ReLU())
        self.stage_fc = nn.Linear(16, d_latent)
        self.subtype_fc = nn.Linear(16, n_subtypes)

        self.latent_dir = nn.Parameter(torch.ones(1, requires_grad=False))  # 1 for ascending latent (0 is control and 1 is patient), 0 for descending

    def forward(self, X):
        h = self.net(X)

        stage = torch.sigmoid(self.stage_fc(h)) * self.n_biomarkers

        # The subtype output is meant to represent a normalised weightage over possible subtypes
        temperature = 0.05
        subtype_output = self.subtype_fc(h)
        subtype = (torch.exp(subtype_output / temperature) /
                   torch.sum(torch.exp(subtype_output / temperature), dim=1).unsqueeze(1))

        return torch.concatenate([stage, subtype], dim=-1)


class Decoder(nn.Module):
    def __init__(self, d_out, d_latent, n_subtypes):
        super().__init__()
        self.d_latent = d_latent
        self.n_subtypes = n_subtypes
        self.net = nn.Sequential(nn.Linear(self.d_latent + self.n_subtypes, 16),
                                 nn.ReLU())

        self.fc = nn.Linear(16, d_out)

    def forward(self, Z):
        h = self.net(Z)

        return self.fc(h)


class AE(ae_stager.AE):
    def __init__(self, enc, dec):
        super().__init__(enc, dec)

    def predict_uncorrected_stage(self, X):
        return self.enc(X)[:, 0].unsqueeze(1) / X.shape[1]  # In the encoder we scale up to distinguish from subtype

    def predict_subtype(self, X):
        return torch.argmax(self.enc(X)[:, 1:], dim=-1)

    def subtype_scores(self, X):
        return self.enc(X)[:, 1:]


# def ae_criterion(X, ae, device):
#     reconstructions = ae.reconstruct_input(X)
#
#     ms_error = ((X - reconstructions) ** 2).mean()
#
#     return ms_error


def run_training(n_epochs, net, model_name, train_loader, val_loader, optimiser, criterion, model_type, device):
    train_dataset_size = len(train_loader.dataset)
    model_dir = os.path.join(SAVED_MODEL_DIR, model_type)
    os.makedirs(model_dir, exist_ok=True)

    enc_path = os.path.join(model_dir, "enc_" + model_name + ".pth")
    dec_path = os.path.join(model_dir, "dec_" + model_name + ".pth")

    epochs_without_improvement = 0
    best_loss = float('inf')
    epoch_patience = 10
    minimum_improvement = 1e-4  # Minimum improvement considered 'significant'

    for epoch in tqdm(range(n_epochs), desc=f"Training {model_type}"):
        train_loss = 0.0
        for (X, _) in train_loader:
            X = X.to(device)
            optimiser.zero_grad()

            loss = criterion(X, net, device)
            loss.backward()

            optimiser.step()
            train_loss += loss.item() * X.shape[0] / train_dataset_size

        # Compute loss on validation set
        val_loss = 0.0
        val_dataset_size = len(val_loader.dataset)
        with torch.no_grad():
            for (X, _) in val_loader:
                X = X.to(device)
                optimiser.zero_grad()

                val_loss += criterion(X, net, device).item() * X.shape[0] / val_dataset_size

        # Compute error between latents and stages, just to track progress
        # rmse_stage_error, reconstruction_error = evaluate_autoencoder(train_loader, net, device)

        if best_loss - val_loss >= minimum_improvement:
            best_loss = val_loss
            torch.save(net.enc.state_dict(), enc_path)
            torch.save(net.dec.state_dict(), dec_path)
            epochs_without_improvement = 0
        else:
            # Terminate training early if no significant improvement
            epochs_without_improvement += 1

            if epochs_without_improvement >= epoch_patience:
                print(
                    f"Ending training early as no significant validation loss decrease "
                    f"has been observed in {epoch_patience} epochs")
                break

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, train loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

    # Load the best model and call compute_latent_direction then store it again before returning
    net.enc.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    net.dec.load_state_dict(torch.load(dec_path, map_location=DEVICE))

    net.automatically_set_latent_direction(train_loader)

    torch.save(net.enc.state_dict(), enc_path)
    torch.save(net.dec.state_dict(), dec_path)


if __name__ == "__main__":
    num_sets = 3
    dataset_names = [f"synthetic_120_10_{i}" for i in range(num_sets)]
    model_name = "synthetic_120_10_0"

    train_set = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TRAIN_DIR, label_directory=SIMULATED_LABEL_TRAIN_DIR)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)

    val_set = SyntheticDatasetVec(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_VAL_DIR, label_directory=SIMULATED_LABEL_VAL_DIR)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True)

    num_biomarkers = next(iter(train_loader))[0].shape[1]

    n_epochs = 1000

    # ---------- Train AE -----------
    ae_enc = Encoder(n_biomarkers=num_biomarkers, d_latent=1, n_subtypes=num_sets).to(DEVICE)
    ae_dec = Decoder(d_out=num_biomarkers, d_latent=1, n_subtypes=num_sets).to(DEVICE)

    ae = AE(enc=ae_enc, dec=ae_dec)

    opt_ae = torch.optim.Adam(itertools.chain(ae_enc.parameters(), ae_dec.parameters()), lr=0.001)

    # Run training loop
    run_training(n_epochs, ae, model_name, train_loader, val_loader,
                 opt_ae, ae_stager.ae_criterion, model_type="ae", device=DEVICE)

    # EVALUATION ---------------------------------------------------------
    # Evaluate on the final models saved during training
    ae_enc_model_path = os.path.join(SAVED_MODEL_DIR, "ae", "enc_" + model_name + ".pth")
    ae_dec_model_path = os.path.join(SAVED_MODEL_DIR, "ae", "dec_" + model_name + ".pth")

    ae_enc.load_state_dict(torch.load(ae_enc_model_path, map_location=DEVICE))
    ae_dec.load_state_dict(torch.load(ae_dec_model_path, map_location=DEVICE))

    staging_mse, reconstruction_mse = evaluation.evaluate_autoencoder(val_loader, ae, DEVICE)
    reconstruction_mse = reconstruction_mse.cpu()

    print("Staging RMSE of trained AE on validation set:", staging_mse)
    print("Reconstruction MSE of trained AE on validation set:", reconstruction_mse)

    # Compute subtype accuracy
    subtype_accuracy = compute_subtype_accuracy_with_cluster_mapping(ae, dataset_names)
    print("Subtype accuracy:", subtype_accuracy)

    # Visualise stage predictions against ground truth
    fig, ax = predicted_stage_comparison(train_loader, num_biomarkers, ae, DEVICE)
    fig.show()

    # Visualise the training set encoded in the latent space
    fig, ax = plot_dataset_in_latent_space(net=ae, dataset_names=dataset_names)
    fig.show()

    plt.show()


