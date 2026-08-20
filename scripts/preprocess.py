from pathlib import Path

import cv2

from src.preprocessing import (
    EXTS,
    CLAHE_CLIP,
    CLAHE_TILE,
    green_clahe_minmax_from_bgr,
)


# Input and output directories
IN_DIR = Path(
    r"D:\Research\CVD\CVD_Project\Fundus_CIMT_2903 Dataset"
)

OUT_DIR = Path(
    r"D:\Research\CVD\CVD_Project\Fundus_green_CLAHE"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Create CLAHE operator
clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP,
    tileGridSize=CLAHE_TILE,
)


# Collect input images recursively
in_files = sorted(
    [
        p
        for p in IN_DIR.rglob("*")
        if p.suffix.lower() in EXTS
    ]
)


# Process each fundus image
for src in in_files:

    # Preserve the input directory structure
    rel = src.relative_to(IN_DIR)
    dst = OUT_DIR / rel

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Read original fundus image
    bgr = cv2.imread(str(src))

    if bgr is None:
        continue

    # Green channel -> CLAHE -> min-max normalization
    g8 = green_clahe_minmax_from_bgr(
        bgr,
        clahe,
    )

    # Save processed image
    cv2.imwrite(
        str(dst),
        g8,
    )


print(
    f"Saved processed images to: {OUT_DIR}"
)