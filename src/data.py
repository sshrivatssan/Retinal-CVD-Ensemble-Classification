import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------
# CNN image caching
# ---------------------------------------------------------------------

def build_original_index(orig_dir: Path):
    """Index original fundus images by filename stem."""

    orig_by_stem = {}

    for root, _, files in os.walk(orig_dir):
        for f in files:
            p = Path(root) / f

            if p.suffix.lower() in VALID_EXT:
                orig_by_stem[p.stem] = p

    return orig_by_stem


def cache_one(
    in_path: Path,
    cache_dir: Path,
    image_size: int,
    clahe,
):
    """Preprocess and cache one fundus image as a uint8 tensor."""

    out_path = cache_dir / f"{in_path.stem}.pt"

    if out_path.exists():
        return True

    img = cv2.imread(
        str(in_path),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return False

    # Green channel
    g = img[:, :, 1]

    # CLAHE enhancement
    g = clahe.apply(g)

    # Resize for CNN input
    g = cv2.resize(
        g,
        (image_size, image_size),
        cv2.INTER_AREA
    )

    # Store as [1, H, W] uint8 tensor
    t = torch.from_numpy(g).to(torch.uint8).unsqueeze(0)

    torch.save(t, out_path)

    return True


def cache_images(
    orig_dir: Path,
    cache_dir: Path,
    image_size: int = 224,
):
    """Cache all available fundus images as preprocessed tensors."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    orig_by_stem = build_original_index(orig_dir)

    made = 0

    for stem, p in orig_by_stem.items():
        if not (cache_dir / f"{stem}.pt").exists():
            if cache_one(
                p,
                cache_dir,
                image_size,
                clahe
            ):
                made += 1

    print(
        f"[CACHE] new tensors made: {made} | "
        f"total cached: {len(list(cache_dir.glob('*.pt')))}"
    )

    return orig_by_stem


# ---------------------------------------------------------------------
# Patient-wise train/validation/test split
# ---------------------------------------------------------------------

def _derive_patient_from_id(s: str) -> str:
    """Derive patient identifier from an image ID."""

    stem = Path(s).stem
    ss = stem.lower()

    # Remove eye/side tokens
    ss = re.sub(
        r'(_|-)?(od|os|ou|le|re|left|right|l|r)\b',
        '',
        ss
    )

    ss = re.sub(
        r'(_|-)?(lt|rt)\b',
        '',
        ss
    )

    ss = re.sub(
        r'(_|-)?(eye|osd|ird|fundus)\b',
        '',
        ss
    )

    # Remove repeated separators
    ss = re.sub(
        r'[_\-]+',
        '_',
        ss
    ).strip('_-')

    return ss


def create_patient_splits(
    json_path: Path,
    cache_dir: Path,
    split_dir: Path,
    orig_by_stem,
    image_size: int = 224,
    seed: int = 42,
):
    """Create patient-wise train, validation, and test CSV files."""

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Load wide JSON and convert to long format
    meta_wide = pd.read_json(json_path)

    meta = (
        meta_wide
        .transpose()
        .reset_index()
        .rename(columns={"index": "id"})
    )

    meta["id"] = meta["id"].astype(str)
    meta["label"] = meta["label"].astype(int)

    # Use explicit patient column when available
    if "patient" in meta.columns:
        meta["patient"] = meta["patient"].astype(str)

    else:
        meta["patient"] = meta["id"].map(
            _derive_patient_from_id
        )

    # Index cached tensors
    cache_by_stem = {
        p.stem: str(p)
        for p in cache_dir.glob("*.pt")
    }

    def resolve_id_to_pt(id_str: str):
        """Match an image ID to its cached tensor."""

        if id_str in cache_by_stem:
            return cache_by_stem[id_str]

        src = orig_by_stem.get(
            Path(id_str).stem,
            None
        )

        if src is not None and cache_one(
            src,
            cache_dir,
            image_size,
            clahe
        ):
            path = str(
                cache_dir / f"{Path(id_str).stem}.pt"
            )

            cache_by_stem[
                Path(id_str).stem
            ] = path

            return path

        # Fuzzy fallback
        cands = [
            s for s in orig_by_stem
            if (
                s == Path(id_str).stem
                or s.endswith(Path(id_str).stem)
                or s.startswith(Path(id_str).stem)
                or Path(id_str).stem in s
            )
        ]

        if cands:
            cands.sort(
                key=lambda s: (len(s), s)
            )

            src = orig_by_stem[cands[0]]

            if cache_one(
                src,
                cache_dir,
                image_size,
                clahe
            ):
                path = str(
                    cache_dir / f"{src.stem}.pt"
                )

                cache_by_stem[src.stem] = path

                return path

        return None

    meta["filepath"] = meta["id"].map(
        resolve_id_to_pt
    )

    missing = meta["filepath"].isna()

    if missing.any():
        ex = (
            meta.loc[missing, "id"]
            .head()
            .tolist()
        )

        raise FileNotFoundError(
            f"{missing.sum()} IDs couldn’t be matched/cached. "
            f"Examples: {ex}"
        )

    # One row per patient
    pat_tab = (
        meta.groupby("patient")
        .agg(
            label=("label", "max"),
            n=("id", "count")
        )
        .reset_index()
    )

    # First split: test patients
    sgkf_test = StratifiedGroupKFold(
        n_splits=int(1 / 0.15),
        shuffle=True,
        random_state=seed
    )

    y_all = pat_tab["label"].to_numpy()
    g_all = pat_tab["patient"].to_numpy()

    test_idx_fold = None

    for _, test_idx in sgkf_test.split(
        np.zeros_like(y_all),
        y_all,
        groups=g_all
    ):
        test_idx_fold = test_idx
        break

    mask_test = np.zeros(
        len(pat_tab),
        dtype=bool
    )

    mask_test[test_idx_fold] = True

    pat_test = (
        pat_tab[mask_test]
        .reset_index(drop=True)
    )

    pat_dev = (
        pat_tab[~mask_test]
        .reset_index(drop=True)
    )

    # Second split: validation patients
    val_target = int(
        round(0.15 * len(pat_tab))
    )

    sgkf_val = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    best_val_idx = None
    best_diff = 10**9

    y_dev = pat_dev["label"].to_numpy()
    g_dev = pat_dev["patient"].to_numpy()

    for _, val_idx in sgkf_val.split(
        np.zeros_like(y_dev),
        y_dev,
        groups=g_dev
    ):
        diff = abs(
            len(val_idx) - val_target
        )

        if diff < best_diff:
            best_diff = diff
            best_val_idx = val_idx

    mask_val = np.zeros(
        len(pat_dev),
        dtype=bool
    )

    mask_val[best_val_idx] = True

    pat_val = (
        pat_dev[mask_val]
        .reset_index(drop=True)
    )

    pat_trn = (
        pat_dev[~mask_val]
        .reset_index(drop=True)
    )

    print(
        f"[patients] total={len(pat_tab)} | "
        f"train={len(pat_trn)} "
        f"val={len(pat_val)} "
        f"test={len(pat_test)}"
    )

    # Map patient splits back to image rows
    def rows_for(patients):
        return (
            meta.loc[
                meta["patient"].isin(
                    patients["patient"]
                ),
                ["filepath", "label", "id"]
            ]
            .rename(
                columns={"id": "filename"}
            )
        )

    tr_df = rows_for(
        pat_trn
    ).reset_index(drop=True)

    va_df = rows_for(
        pat_val
    ).reset_index(drop=True)

    te_df = rows_for(
        pat_test
    ).reset_index(drop=True)

    split_dir.mkdir(parents=True, exist_ok=True)

    tr_df.to_csv(
        split_dir / "train.csv",
        index=False
    )

    va_df.to_csv(
        split_dir / "val.csv",
        index=False
    )

    te_df.to_csv(
        split_dir / "test.csv",
        index=False
    )

    def _counts(df):
        return {
            "n": len(df),
            "labels": df[
                "label"
            ].value_counts().to_dict()
        }

    print(
        "✅ SPLITS (patient-wise 70/15/15) READY"
    )

    print(
        "train:",
        _counts(tr_df)
    )

    print(
        "val:  ",
        _counts(va_df)
    )

    print(
        "test: ",
        _counts(te_df)
    )

    return tr_df, va_df, te_df


# ---------------------------------------------------------------------
# Split loading and GPU tensors
# ---------------------------------------------------------------------

def load_split_csv(csv_path: Path):
    """Load filepaths, labels, and filenames from a split CSV."""

    df = pd.read_csv(csv_path)

    return (
        df["filepath"].tolist(),
        df["label"].astype(int).to_numpy(),
        df["filename"].astype(str).tolist()
    )


def stack_to_gpu(
    paths,
    device,
    mean,
    std,
):
    """Load cached tensors, normalize them, and stack them on the GPU."""

    xs = []

    for p in paths:

        # Load uint8 [1,H,W] tensor
        t = torch.load(
            p,
            map_location="cpu"
        )

        if t.ndim == 2:
            t = t.unsqueeze(0)

        # Convert to float and scale to [0,1]
        t = (
            t.to(
                device,
                non_blocking=True
            )
            .float()
            .div_(255.0)
        )

        # Convert single channel to three channels
        if t.size(0) == 1:
            t = t.repeat(
                3,
                1,
                1
            )

        # Normalize
        t = (
            t - mean[:, None, None]
        ) / std[:, None, None]

        # Store as FP16
        xs.append(
            t.half()
        )

    return torch.stack(
        xs,
        dim=0
    )


def load_split_tensors(
    split_dir: Path,
    device,
    mean,
    std,
):
    """Load train, validation, and test splits into tensors."""

    tr_paths, tr_labels, tr_names = load_split_csv(
        split_dir / "train.csv"
    )

    va_paths, va_labels, va_names = load_split_csv(
        split_dir / "val.csv"
    )

    te_paths, te_labels, te_names = load_split_csv(
        split_dir / "test.csv"
    )

    print("→ Loading to GPU …")

    Xtr = stack_to_gpu(
        tr_paths,
        device,
        mean,
        std
    )

    Xva = stack_to_gpu(
        va_paths,
        device,
        mean,
        std
    )

    Xte = stack_to_gpu(
        te_paths,
        device,
        mean,
        std
    )

    Ytr = torch.from_numpy(
        tr_labels
    ).to(device).long()

    Yva = torch.from_numpy(
        va_labels
    ).to(device).long()

    Yte = torch.from_numpy(
        te_labels
    ).to(device).long()

    if device.type == "cuda":
        print(
            "GPU tensors:",
            Xtr.shape,
            Xva.shape,
            Xte.shape,
            "| mem≈",
            f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
        )

    else:
        print(
            "Tensors on CPU:",
            Xtr.shape,
            Xva.shape,
            Xte.shape
        )

    return (
        Xtr,
        Ytr,
        Xva,
        Yva,
        Xte,
        Yte,
        tr_names,
        va_names,
        te_names
    )


# ---------------------------------------------------------------------
# Handcrafted feature alignment
# ---------------------------------------------------------------------

def _align(ids, df, id_col):
    """Align handcrafted-feature rows with image IDs."""

    key = (
        df[id_col]
        .astype(str)
        .str.replace(
            "\\",
            "/",
            regex=False
        )
    )

    base = (
        key
        .str.split("/")
        .str[-1]
    )

    stem = base.str.replace(
        r"\.[^.]+$",
        "",
        regex=True
    )

    i_full = dict(
        zip(key, df.index)
    )

    i_base = dict(
        zip(base, df.index)
    )

    i_stem = dict(
        zip(stem, df.index)
    )

    rows = []
    miss = []

    for k in ids:

        kb = k.split("/")[-1]
        ks = kb.rsplit(".", 1)[0]

        if k in i_full:
            rows.append(
                df.loc[i_full[k]]
            )

        elif kb in i_base:
            rows.append(
                df.loc[i_base[kb]]
            )

        elif ks in i_stem:
            rows.append(
                df.loc[i_stem[ks]]
            )

        else:
            miss.append(k)

    if miss:
        raise ValueError(
            f"Missing handcrafted rows for "
            f"{len(miss)} ids, e.g. {miss[:8]}"
        )

    return pd.DataFrame(
        rows
    ).reset_index(drop=True)


def _numeric_features(df, id_col="filename"):
    """Retain numeric handcrafted feature columns."""

    return (
        df.drop(
            columns=[
                id_col,
                "label",
                "patient_id",
                "eye"
            ],
            errors="ignore"
        )
        .select_dtypes(
            include=[np.number]
        )
    )


def prepare_handcrafted_features(
    hand_feats: Path,
    tr_df,
    va_df,
    te_df,
    Xtr,
    Xva,
    Xte,
    device,
):
    """Align, standardize, and convert handcrafted features to tensors."""

    # IDs from cached tensor filenames
    IDS_tr = (
        tr_df["filepath"]
        .astype(str)
        .apply(
            lambda s: Path(s).name
        )
        .tolist()
    )

    IDS_va = (
        va_df["filepath"]
        .astype(str)
        .apply(
            lambda s: Path(s).name
        )
        .tolist()
    )

    IDS_te = (
        te_df["filepath"]
        .astype(str)
        .apply(
            lambda s: Path(s).name
        )
        .tolist()
    )

    assert len(IDS_tr) == Xtr.size(0), (
        "Train split rows != train tensor rows."
    )

    assert len(IDS_va) == Xva.size(0), (
        "Val split rows != val tensor rows."
    )

    assert len(IDS_te) == Xte.size(0), (
        "Test split rows != test tensor rows."
    )

    # Load handcrafted features
    df_h = pd.read_csv(
        str(hand_feats)
    )

    ID_COL = "filename"

    HF_tr = _align(
        IDS_tr,
        df_h,
        ID_COL
    )

    HF_va = _align(
        IDS_va,
        df_h,
        ID_COL
    )

    HF_te = _align(
        IDS_te,
        df_h,
        ID_COL
    )

    # Keep numeric feature columns
    HF_tr_np = _numeric_features(
        HF_tr,
        ID_COL
    ).to_numpy(np.float32)

    HF_va_np = _numeric_features(
        HF_va,
        ID_COL
    ).to_numpy(np.float32)

    HF_te_np = _numeric_features(
        HF_te,
        ID_COL
    ).to_numpy(np.float32)

    # Fit scaler using training features only
    hf_scaler = StandardScaler().fit(
        HF_tr_np
    )

    HFtr_s = hf_scaler.transform(
        HF_tr_np
    )

    HFva_s = hf_scaler.transform(
        HF_va_np
    )

    HFte_s = hf_scaler.transform(
        HF_te_np
    )

    # Convert to GPU tensors
    HFtr_t = torch.tensor(
        HFtr_s,
        dtype=torch.float32,
        device=device
    )

    HFva_t = torch.tensor(
        HFva_s,
        dtype=torch.float32,
        device=device
    )

    HFte_t = torch.tensor(
        HFte_s,
        dtype=torch.float32,
        device=device
    )

    HF_DIM = HFtr_t.size(1)

    print(
        f"Handcrafted dims -> "
        f"TR {HFtr_t.shape} | "
        f"VA {HFva_t.shape} | "
        f"TE {HFte_t.shape}"
    )

    return (
        HFtr_t,
        HFva_t,
        HFte_t,
        HF_DIM,
        hf_scaler
    )