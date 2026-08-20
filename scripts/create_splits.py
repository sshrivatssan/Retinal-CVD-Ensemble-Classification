from pathlib import Path

from src.data import (
    cache_images,
    create_patient_splits,
)


# Dataset paths
ORIG_DIR = Path(
    r"D:\Research\CVD\CVD_Project\Fundus_CIMT_2903 Dataset"
)

JSON_PATH = Path(
    r"D:\Research\CVD\CVD_Project\data_info.json"
)

CACHE_DIR = Path(
    "_validation_runs/Fundus_green_CLAHE_pt"
)

SPLIT_DIR = Path(
    "_validation_runs/CVD_split"
)


# Experimental settings
IMG_SIZE = 224
SEED = 42


# Create cached tensors
orig_by_stem = cache_images(
    orig_dir=ORIG_DIR,
    cache_dir=CACHE_DIR,
    image_size=IMG_SIZE,
)


# Create patient-wise train/validation/test splits
tr_df, va_df, te_df = create_patient_splits(
    json_path=JSON_PATH,
    cache_dir=CACHE_DIR,
    split_dir=SPLIT_DIR,
    orig_by_stem=orig_by_stem,
    image_size=IMG_SIZE,
    seed=SEED,
)