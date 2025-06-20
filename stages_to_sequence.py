import torch
import os
import json
import numpy as np

from config import DEVICE, MODEL_DIR, SIMULATED_LABELS_DIR, SIMULATED_OBSERVATIONS_DIR
import vae_stager
import ae_stager
from datasets.synthetic_dataset_vector import SyntheticDatasetVec
from evaluation import evaluate_autoencoder, evaluate_sequence


def stages_to_sequence(num_biomarkers, dataloader, net, device):
    # Infer underlying sequence from latent stage predictions.
    # By this formula, a later score implies earlier abnormality and thus earlier position.
    # But it still depends on the direction of the learned latent...
    biomarker_scores = torch.zeros(num_biomarkers, device=device, requires_grad=False)

    start = torch.zeros(num_biomarkers, device=device, requires_grad=False)  # estimate of biomarker levels at the start
    end = torch.zeros(num_biomarkers, device=device, requires_grad=False)  # estimate of biomarker levels at the end

    start_count = 0
    end_count = 0

    for X, _ in dataloader:
        X = X.to(device)
        pred = net.encode(X)

        # consider measurements predicted to be within stages 0 to 3 and num_biomarkers - 3 to num_biomarkers?
        idx_within_start = torch.where(pred < 3 / num_biomarkers)[0]  # extract single tensor from tuple
        idx_within_end = torch.where(pred > 1 - (3 / num_biomarkers))[0]

        start_count += idx_within_start.shape[0]
        end_count += idx_within_end.shape[0]

        if start_count > 0:
            start = start * (start_count - idx_within_start.shape[0]) / start_count \
                + X[idx_within_start].sum(0) / start_count
        if end_count > 0:
            end = end * (end_count - idx_within_end.shape[0]) / end_count \
                + X[idx_within_end].sum(0) / end_count

    reverse = torch.ones(num_biomarkers).to(device)
    reverse[torch.where(start > end)] = -1  # indicating that the level decreases with abnormality

    midpoints = start + (end - start) / 2

    corrected_start = torch.min(start, end)
    corrected_end = torch.max(start, end)

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


if __name__ == "__main__":
    dataset_name = "synthetic_120_10_dpm_0"

    # Load dataset
    dataset = SyntheticDatasetVec(dataset_name=dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    example_x, _ = next(iter(dataloader))
    num_biomarkers = example_x.shape[1]

    # Read in ground truth sequence
    seq_label_file = os.path.join(SIMULATED_LABELS_DIR, dataset_name + "_seq.json")
    with open(seq_label_file, 'r') as f:
        seq_gt = json.load(f)

    # flatten sequence
    seq_gt = np.array(seq_gt).squeeze(1)

    print("gt sequence:\n", seq_gt)

    # -------- VAE ----------
    vae_enc = vae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    vae_dec = vae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

    vae = vae_stager.VAE(enc=vae_enc, dec=vae_dec)

    # Load a model fitted to the particular dataset
    enc_model_path = os.path.join(MODEL_DIR, "vae", "enc_" + dataset_name + ".pth")
    dec_model_path = os.path.join(MODEL_DIR, "vae", "dec_" + dataset_name + ".pth")

    vae_enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
    vae_dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

    # # delete later. verify that everything loaded alright
    print("Mean squared error:", evaluate_autoencoder(dataloader, vae, DEVICE)[0])

    # Infer sequence from stage information
    vae_seq_prediction = stages_to_sequence(num_biomarkers, dataloader, vae, DEVICE)

    print("seq inferred from VAE predictions:")
    print(vae_seq_prediction)
    print("VAE sequence score:", evaluate_sequence(vae_seq_prediction.cpu(), seq_gt))

    # --------- AE ----------
    ae_enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
    ae_dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

    ae = ae_stager.AE(enc=ae_enc, dec=ae_dec)

    # Load a model fitted to the particular dataset
    enc_model_path = os.path.join(MODEL_DIR, "ae", "enc_" + dataset_name + ".pth")
    dec_model_path = os.path.join(MODEL_DIR, "ae", "dec_" + dataset_name + ".pth")

    ae_enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
    ae_dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

    # # delete later. verify that everything loaded alright
    print("Mean squared error:", evaluate_autoencoder(dataloader, ae, DEVICE)[0])

    # Infer sequence from stage information
    ae_seq_prediction = stages_to_sequence(num_biomarkers, dataloader, ae, DEVICE)

    print("seq inferred from AE predictions:")
    print(ae_seq_prediction)
    print("AE sequence score:", evaluate_sequence(ae_seq_prediction.cpu(), seq_gt))

