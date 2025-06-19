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
    # The problem is it assumes an identical normal and abnormal value for all biomarkers.
    # Maybe first obtain biomarker values at the min and max stages to rescale into 0 normal and 1 abnormal?
    biomarker_scores = torch.zeros(num_biomarkers, device=device, requires_grad=False)

    for X, _ in dataloader:
        X = X.to(device)
        pred = net.encode(X)
        biomarker_scores += (pred * X).sum(dim=0)

    return torch.argsort(biomarker_scores, descending=True)


if __name__ == "__main__":
    dataset_name = "synthetic_1200_50_dpm_0"

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
    # print("Mean squared error:", evaluate_autoencoder(dataloader, vae, DEVICE)[0])

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
    # print("Mean squared error:", evaluate_autoencoder(dataloader, ae, DEVICE)[0])

    # Infer sequence from stage information
    ae_seq_prediction = stages_to_sequence(num_biomarkers, dataloader, ae, DEVICE)

    print("seq inferred from AE predictions:")
    print(ae_seq_prediction)
    print("AE sequence score:", evaluate_sequence(ae_seq_prediction.cpu(), seq_gt))

