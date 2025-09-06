import os

import torch
import itertools
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import DEVICE, SAVED_MODEL_DIR, SIMULATED_OBS_TRAIN_DIR, SIMULATED_LABEL_TRAIN_DIR, ADNI_DIR
from datasets.synthetic_dataset import SyntheticDataset
from evaluation import evaluate_autoencoder

from models import ae_stager, vae_stager


def run_training(n_epochs, net, model_name, train_loader, val_loader, optimiser, criterion, model_type, device,
                 min_epochs=0):
    train_dataset_size = len(train_loader.dataset)
    model_dir = os.path.join(SAVED_MODEL_DIR, model_type)
    os.makedirs(model_dir, exist_ok=True)

    enc_path = os.path.join(model_dir, "enc_" + model_name + ".pth")
    dec_path = os.path.join(model_dir, "dec_" + model_name + ".pth")

    epochs_without_improvement = 0
    best_loss = float('inf')
    epoch_patience = 10
    minimum_improvement = 0  # Minimum improvement considered 'significant'

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

        if best_loss - val_loss >= minimum_improvement:
            best_loss = val_loss
            torch.save(net.enc.state_dict(), enc_path)
            torch.save(net.dec.state_dict(), dec_path)
            epochs_without_improvement = 0
        elif epoch > min_epochs:  # only start counting once the minimum number of epochs has passed
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
    # USER CONFIGURATION --------------------
    num_sets = 1
    # dataset_names = [f"synthetic_1200_100_{i}" for i in range(num_sets)]
    # model_name = "synthetic_120_10_0"
    # dataset_names = ["adni_longitudinal_data_all_volumes_abpos"]
    # model_name = dataset_names[0]
    dataset_names = ["rexample_0"]
    model_name = dataset_names[0]

    model_type = "ae"  # only vae or ae supported currently
    if model_type not in ["vae", "ae"]:
        print("Model type must be one of 'vae' or 'ae' (case-sensitive)")
        exit()

    dataset_type = "synthetic"  # only 'synthetic' or 'adni' supported currently
    if dataset_type not in ["synthetic", "adni"]:
        print("Dataset type must be one of 'synthetic' or 'adni' (case-sensitive)")
        exit()
    # ---------------------------------------

    # Load data
    dataset = None
    if dataset_type == "synthetic":
        dataset = SyntheticDataset(dataset_names=dataset_names, obs_directory=SIMULATED_OBS_TRAIN_DIR,
                                   label_directory=SIMULATED_LABEL_TRAIN_DIR)
    elif dataset_type == "adni":
        dataset = SyntheticDataset(dataset_names=dataset_names, obs_directory=ADNI_DIR)

    # Split training set
    train_indices, val_indices = train_test_split(range(len(dataset)), train_size=0.8)

    # Generate subset based on indices
    train_split = torch.utils.data.Subset(dataset, train_indices)
    val_split = torch.utils.data.Subset(dataset, val_indices)

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=16, shuffle=True)

    num_biomarkers = next(iter(train_loader))[0].shape[1]

    n_epochs = 1000

    # Instantiate network
    enc = None
    dec = None

    if model_type == "vae":
        enc = vae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
        dec = vae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

        net = vae_stager.VAE(enc=enc, dec=dec)
        criterion = vae_stager.vae_criterion_wrapper(beta=1)

    elif model_type == "ae":
        enc = ae_stager.Encoder(d_in=num_biomarkers, d_latent=1).to(DEVICE)
        dec = ae_stager.Decoder(d_out=num_biomarkers, d_latent=1).to(DEVICE)

        net = ae_stager.AE(enc=enc, dec=dec)
        criterion = ae_stager.ae_criterion

    if enc is None or dec is None:
        print("Arguments not validated properly - please fix")
        exit()

    opt = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=0.001)

    # Run training loop
    run_training(n_epochs, net, model_name, train_loader, val_loader, opt,
                 criterion, model_type=model_type, device=DEVICE)

    # Evaluate on the final models saved during training
    enc_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "enc_" + model_name + ".pth")
    dec_model_path = os.path.join(SAVED_MODEL_DIR, model_type, "dec_" + model_name + ".pth")

    enc.load_state_dict(torch.load(enc_model_path, map_location=DEVICE))
    dec.load_state_dict(torch.load(dec_model_path, map_location=DEVICE))

    staging_rmse, reconstruction_mse = evaluate_autoencoder(val_loader, net, DEVICE)
    reconstruction_mse = reconstruction_mse.cpu()

    # This check only exists because the processed ADNI data I am using has no labels currently
    if dataset_type == "synthetic":
        print("Staging RMSE on validation set:", staging_rmse)

    print("Reconstruction MSE on validation set:", reconstruction_mse)
