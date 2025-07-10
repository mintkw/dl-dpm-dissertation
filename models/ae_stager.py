import os

import torch
import torch.nn as nn
import itertools
from tqdm import tqdm

from config import DEVICE, SAVED_MODEL_DIR
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


def ae_criterion(X, ae, device):
    reconstructions = ae.reconstruct(X)

    latents = ae.encode(X)

    ms_error = ((X - reconstructions) ** 2).mean()

    # use mean correlation as a way to regularise the direction of the latent (lower for lower biomarker values)
    data_matrix = torch.concat([X, latents], dim=1)  # columns as variables and rows as observations

    # EXPERIMENT ---------------
    # PEARSON COEFF
    # correlation_matrix = torch.corrcoef(torch.transpose(data_matrix, 1, 0))
    # mean_correlation = correlation_matrix[-1][:-1].mean()

    # OR SPEARMAN COEFF
    # correlation_matrix = stats.spearmanr(data_matrix.detach().cpu()).statistic
    # mean_correlation = correlation_matrix[-1][:-1].mean()

    # if np.any(np.isnan(correlation_matrix[-1])):
    #     print("correlation matrix")
    #     exit()
    # if np.isnan(mean_correlation):
    #     print("mean correlation")
    #     exit()
    #
    # return rms_error - mean_correlation
    # --------------------------

    # if torch.any(torch.isnan(reconstructions)):
    #     print("reconstruction")
    #     exit()
    # if torch.any(torch.isnan(latents)):
    #     print("latents")
    #     exit()
    # if torch.any(torch.isnan(ms_error)):
    #     print("ms error")
    #     exit()
    # if torch.any(torch.isnan(data_matrix)):
    #     print("data matrix")
    #     exit()

    return ms_error


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
