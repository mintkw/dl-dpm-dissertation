import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Directories
DATA_DIR = "data"
SIMULATED_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
SIMULATED_OBSERVATIONS_DIR = os.path.join(SIMULATED_DATA_DIR, "observations")
SIMULATED_LABELS_DIR = os.path.join(SIMULATED_DATA_DIR, "labels")
MODEL_DIR = "models"
VAE_MODEL_DIR = os.path.join(MODEL_DIR, "vae")