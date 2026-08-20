import json
import os
from pathlib import Path

import cv2
import pandas as pd

from src.preprocessing import make_mask_from_processed
from src.features import feats_lbp, feats_glcm


# Dataset paths
JSON_PATH = Path(
    r"D:\Research\CVD\CVD_Project\data_info.json"
)

PROC_DIR = Path(
    r"D:\Research\CVD\CVD_Project\Fundus_green_CLAHE"
)

OUT_CSV = Path(
    r"D:\Research\CVD\CVD_Project\features_lbp_glcm_all.csv"
)


# Supported image formats
EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# Load patient metadata
with open(
    JSON_PATH,
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

assert isinstance(
    data,
    dict,
), "data_info.json must be dict: patient_id -> {...}"


rows = []
missing_proc = []


# Extract features from each available eye image
for pid, info in data.items():

    label = int(
        info.get("label", 0)
    )

    for eye_key, eye_tag in (
        ("left_eye", "L"),
        ("right_eye", "R"),
    ):

        fn = info.get(eye_key)

        if not fn:
            continue

        fname = os.path.basename(fn)
        p_proc = PROC_DIR / fname

        if (
            p_proc.suffix.lower() not in EXTS
            or not p_proc.exists()
        ):
            missing_proc.append(fname)
            continue

        # Read processed grayscale image
        g8 = cv2.imread(
            str(p_proc),
            cv2.IMREAD_GRAYSCALE,
        )

        if g8 is None:
            missing_proc.append(fname)
            continue

        # Generate normalized image and fundus mask
        g_norm, mask = make_mask_from_processed(
            g8
        )

        # Uniform LBP: P=16, R=2
        d_lbp = feats_lbp(
            g_norm,
            mask,
            P=16,
            R=2,
        )

        # GLCM: 64 levels, distances 1 and 2,
        # angles 0°, 45°, 90°, and 135°
        d_glcm = feats_glcm(
            g_norm,
            mask,
            levels=64,
        )

        # Store metadata and extracted features
        row = {
            "patient_id": str(pid),
            "eye": eye_tag,
            "filename": fname,
            "label": label,
        }

        row.update(d_lbp)
        row.update(d_glcm)

        rows.append(row)


# Build and save feature matrix
df = pd.DataFrame(
    rows
).fillna(0.0)

print(
    "Feature matrix shape:",
    df.shape,
)

df.to_csv(
    OUT_CSV,
    index=False,
)

print(
    "Saved CSV:",
    OUT_CSV,
)


# Report missing or unreadable processed images
if missing_proc:

    miss = sorted(
        set(missing_proc)
    )

    print(
        f"WARNING: {len(miss)} processed files "
        "missing or unreadable."
    )

    print(
        "Examples:",
        miss[:10],
    )