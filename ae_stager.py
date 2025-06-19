import os

import torch
import torch.nn as nn
import itertools
from tqdm import tqdm

from config import DEVICE, MODEL_DIR
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from evaluation import evaluate_autoencoder


class Encoder(nn.Module):

    def __init__(self, d_in, d_latent):
        super().__init__()
        self.d_latent = d_latent
        self.net = nn.Sequential(nn.Linear(d_in, 16),
                                 nn.ReLU(),
                                 nn.Linear(16, d_latent),
                                 nn.Sigmoid())

    def forward(self, X):
        h = self.net(X)

        return h


class Decoder(nn.Module):
    def __init__(self, d_out, d_latent):
        super().__init__()
        self.d_latent = d_latent
        self.net = nn.Sequential(nn.Linear(self.d_latent, 16),
                                 nn.ReLU())

        self.fc = nn.Linear(16, d_out)

    def forward(self, Z):
        h = self.net(Z)

        return self.fc(h)


class AE:
    def __init__(self, enc, dec):
        self.enc = enc
        self.dec = dec

    def encode(self, X):
        return self.enc(X)

    def reconstruct(self, X):
        return self.dec(self.enc(X))


def train_ae(N_epochs, net, train_loader, train_dataset, optimiser, dataset_name, device):
    lowest_reconstruction_error = float('inf')
    ae_model_dir = os.path.join(MODEL_DIR, "ae")
    os.makedirs(ae_model_dir, exist_ok=True)

    enc_path = os.path.join(ae_model_dir, "enc_" + dataset_name + ".pth")
    dec_path = os.path.join(ae_model_dir, "dec_" + dataset_name + ".pth")

    for epoch in tqdm(range(N_epochs), desc="Training AE"):
        train_loss = 0.0
        for X, _ in train_loader:
            X = X.to(device)

            optimiser.zero_grad()

            reconstructions = net.reconstruct(X)
            loss = ((X - reconstructions) ** 2).sum()

            loss.backward()
            optimiser.step()
            train_loss += loss.item() * X.shape[0] / len(train_dataset)  # todo: wait is this formula correct for moving average pls advise...

        if epoch % 10 == 0:
            # compute error between latents and stages - just to track progress
            mse_stage_error, reconstruction_error = evaluate_autoencoder(train_loader, net, device)

            if reconstruction_error < lowest_reconstruction_error:
                lowest_reconstruction_error = min(reconstruction_error.item(), lowest_reconstruction_error)
                torch.save(net.enc.state_dict(), enc_path)
                torch.save(net.dec.state_dict(), dec_path)

            print(f"Epoch {epoch}, train loss = {train_loss:.4f}, average reconstruction squared distance = {reconstruction_error:.4f}, MSE stage error = {mse_stage_error:.4f}")


# if __name__ == "__main__":
#     dataset_name = "synthetic_120_10_dpm_0"
#
#     train_set = SyntheticDatasetVec(dataset_name=dataset_name)
#     train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
#
#     example_x, _ = next(iter(train_dataloader))
#     num_biomarkers = example_x.shape[1]
#
#     enc = Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
#     dec = Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)
#
#     opt_ae = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=0.001)
#
#     # Run training loop
#     n_epochs = 50
#     train_ae(n_epochs, enc, dec, train_dataloader, train_set, opt_ae, dataset_name, DEVICE)
#
#     # Evaluate by computing MSE error
#     print("Mean squared error:", evaluate_autoencoder(train_dataloader, enc, dec, DEVICE)[0])
#
#
#
#     # just for illustrative purposes - print labels and predictions
#     predictions = []
#     gt_stages = []
#     with torch.no_grad():
#         for X, label in train_dataloader:
#             gt_stages.append(label)
#
#             X = X.to(DEVICE)
#             raw_preds = enc(X)  # take mean as preds or sample?
#             predictions.append(raw_preds)
#
#     # scale predictions
#     predictions = torch.concatenate(predictions).squeeze().to(DEVICE)
#     # predictions = (predictions - torch.min(predictions)) / (torch.max(predictions) - torch.min(predictions))
#     # predictions = torch.sigmoid(predictions)
#
#     # scale stages
#     gt_stages = torch.concatenate(gt_stages).to(DEVICE)
#     gt_stages /= torch.max(gt_stages)
#
#     # latent can scale in an opposite direction to stages, so try both directions and take the min error
#     mse_stage_error = torch.min(torch.mean((predictions - gt_stages) ** 2),
#                                 torch.mean((predictions - 1 + gt_stages) ** 2))

    # print(gt_stages)
    # print(predictions)
