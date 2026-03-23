import torch
from builders.model_builder import MODEL_REGISTRY

# --- General Project Configuration ---
class GENERAL:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = torch.cuda.is_available()
    NUM_WORKERS = 4
    SEED = 42
    IMG_SIZE = 320
    NUM_CLASSES = 1
    GPU_IDS = [0] # Default GPU ID(s)

# --- Directory Paths ---
class PATHS:
    LOGS_DIR = "logs"
    RESULTS_DIR = "results"
    CHECKPOINT_DIR = "checkpoints"

# --- Dataset Configurations ---
# For each dataset, provide the base directory and the filenames for each split.
# The mean and std values are crucial for correct normalization.
DATASETS = {
    "AsphaltCrack300": {
        "base_dir": "data/AsphaltCrack300/",
        "train_image_file": "train/asphalt300_train.txt",
        "train_mask_file": "train/asphalt300_train.txt",
        "val_image_file": "val/asphalt300_val.txt",
        "val_mask_file": "val/asphalt300_val.txt",
        "test_image_file": "test/asphalt300_test.txt",
        "test_mask_file": "test/asphalt300_test.txt",
        "mean": [0.425, 0.432, 0.438], "std": [0.232, 0.234, 0.235],
    },
    "CFD": {
        "base_dir": "data/CFD",
        "train_image_file": "train/cfd_train_images.txt",
        "train_mask_file": "train/cfd_train_masks.txt",
        "val_image_file": "val/cfd_val_images.txt",
        "val_mask_file": "val/cfd_val_masks.txt",
        "test_image_file": "test/cfd_test_images.txt",
        "test_mask_file": "test/cfd_test_masks.txt",
        "mean": [0.561, 0.578, 0.589], "std": [0.099, 0.099, 0.099],
    },
    "Crack500": {
        "base_dir": "data/Crack500/",
        "train_image_file": "train/crack500_train.txt",
        "train_mask_file": "train/crack500_train.txt",
        "val_image_file": "val/crack500_val.txt",
        "val_mask_file": "val/crack500_val.txt",
        "test_image_file": "test/crack500_test.txt",
        "test_mask_file": "test/crack500_test.txt",
        "mean": [0.497, 0.496, 0.494], "std": [0.156, 0.159, 0.160],
    },
    "DeepCrack": {
        "base_dir": "data/DeepCrack/",
        "train_image_file": "train/deepcrack_train.txt",
        "train_mask_file": "train/deepcrack_train.txt",
        "val_image_file": "val/deepcrack_val.txt",
        "val_mask_file": "val/deepcrack_val.txt",
        "test_image_file": "test/deepcrack_test.txt",
        "test_mask_file": "test/deepcrack_test.txt",
        "mean": [0.557, 0.574, 0.586], "std": [0.098, 0.098, 0.098],
    },
    "GAPS384": {
        "base_dir": "data/GAPS384/",
        "train_image_file": "train/gaps384_train.txt",
        "train_mask_file": "train/gaps384_train.txt",
        "val_image_file": "val/gaps384_val.txt",
        "val_mask_file": "val/gaps384_val.txt",
        "test_image_file": "test/gaps384_test.txt",
        "test_mask_file": "test/gaps384_test.txt",
        "mean": [0.398, 0.398, 0.398],
        "std": [0.116, 0.116, 0.116],
    },
}

# --- Training Hyperparameters ---
class TRAIN:
    BATCH_SIZE = 8
    EPOCHS = 50
    BASE_LR = 0.0001
    SCHEDULER = "cosine"
    # Use a stable default combined loss that worked well pre-refactor
    LOSS_FN = "BCEDiceLoss"

# --- Testing Configuration ---
class TEST:
    TEST_BATCH_SIZE = 8

# --- Experiment Tracking ---
class TRACKING:
    WANDB_ENABLED = True
    WANDB_PROJECT_NAME = "Crack-Segmentation-Project"

class METRICS:
    # Use a standard default; best threshold is tuned each epoch on validation
    THRESHOLD = 0.5

class LOSS:
    # Historical defaults that balance precision and recall better on CFD
    TVERSKY_ALPHA = 0.7
    TVERSKY_BETA = 0.3

# --- Normalization Options ---
class NORMALIZATION:
    # mode: 'dataset' uses per-dataset stats; 'simple' uses 0.5/0.5 as in legacy code
    MODE = 'simple'

# --- Model Selection ---
# This list is generated automatically from the model registry.
AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())
# Define a smaller subset of models to run by default if '--models' is not specified.
DEFAULT_MODELS_TO_TRAIN = ["U_Net", "SSNet_T", "SSNet_S"]

