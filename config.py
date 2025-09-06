import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SIMULATED_DATA_DIR = os.path.join(ROOT_DIR, DATA_DIR, "synthetic")
SIMULATED_OBS_DIR = os.path.join(SIMULATED_DATA_DIR, "observations")
SIMULATED_LABEL_DIR = os.path.join(SIMULATED_DATA_DIR, "labels")
SIMULATED_OBS_TRAIN_DIR = os.path.join(SIMULATED_OBS_DIR, "train")
SIMULATED_OBS_TEST_DIR = os.path.join(SIMULATED_OBS_DIR, "test")
SIMULATED_LABEL_TRAIN_DIR = os.path.join(SIMULATED_LABEL_DIR, "train")
SIMULATED_LABEL_TEST_DIR = os.path.join(SIMULATED_LABEL_DIR, "test")
SAVED_MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
PLOT_DIR = os.path.join(ROOT_DIR, "plots")
ADNI_DIR = os.path.join(ROOT_DIR, DATA_DIR, "adnimerge")
