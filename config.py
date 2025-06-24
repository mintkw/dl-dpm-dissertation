import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Directories
DATA_DIR = "data"
SIMULATED_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
SIMULATED_OBS_DIR = os.path.join(SIMULATED_DATA_DIR, "observations")
SIMULATED_LABEL_DIR = os.path.join(SIMULATED_DATA_DIR, "labels")
SIMULATED_OBS_TRAIN_DIR = os.path.join(SIMULATED_OBS_DIR, "train")
SIMULATED_OBS_VAL_DIR = os.path.join(SIMULATED_OBS_DIR, "val")
SIMULATED_LABEL_TRAIN_DIR = os.path.join(SIMULATED_LABEL_DIR, "train")
SIMULATED_LABEL_VAL_DIR = os.path.join(SIMULATED_LABEL_DIR, "val")
MODEL_DIR = "models"
PLOT_DIR = "plots"