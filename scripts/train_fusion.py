from datetime import datetime
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from src.data import (
    load_split_tensors,
    prepare_handcrafted_features,
)
from src.models import (
    FusionDenseNet,
    FusionEffB3,
    FusionMobileNetV3,
)
from src.training import (
    LR_RUN,
    EPOCHS_RUN,
    PATIENCE_RUN,
    WEIGHT_DECAY,
    STOP_METRIC,
    create_criterion,
    create_scheduler,
    create_amp,
    epoch_fusion,
    fusion_validation_probabilities,
    tune_threshold,
)

# ---------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    "Device:",
    device,
    "|",
    torch.cuda.get_device_name(0)
    if device.type == "cuda"
    else "CPU",
)

assert device.type == "cuda", (
    "CUDA not available. Install CUDA PyTorch and try again."
)

# ---------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------

SPLIT_DIR = Path(
    "_validation_runs/CVD_split"
)

HAND_FEATS = Path(
    r"D:\Research\CVD\CVD_Project\features_lbp_glcm_all.csv"
)

NUM_CLASSES = 2
BATCH_SIZE = 32

MEAN = torch.tensor(
    [0.5, 0.5, 0.5],
    dtype=torch.float32,
    device=device,
)

STD = torch.tensor(
    [0.25, 0.25, 0.25],
    dtype=torch.float32,
    device=device,
)

# ---------------------------------------------------------------------
# Load image tensors
# ---------------------------------------------------------------------

(
    Xtr,
    Ytr,
    Xva,
    Yva,
    Xte,
    Yte,
    tr_names,
    va_names,
    te_names,
) = load_split_tensors(
    split_dir=SPLIT_DIR,
    device=device,
    mean=MEAN,
    std=STD,
)


# Load split CSVs for handcrafted-feature alignment
tr_df = pd.read_csv(SPLIT_DIR / "train.csv")
va_df = pd.read_csv(SPLIT_DIR / "val.csv")
te_df = pd.read_csv(SPLIT_DIR / "test.csv")


# ---------------------------------------------------------------------
# Prepare handcrafted features
# ---------------------------------------------------------------------

(
    HFtr_t,
    HFva_t,
    HFte_t,
    HF_DIM,
    hf_scaler,
) = prepare_handcrafted_features(
    hand_feats=HAND_FEATS,
    tr_df=tr_df,
    va_df=va_df,
    te_df=te_df,
    Xtr=Xtr,
    Xva=Xva,
    Xte=Xte,
    device=device,
)


# ---------------------------------------------------------------------
# Fusion experiments
# ---------------------------------------------------------------------
EXPERIMENTS = {
    "densenet121_fusion": {
        "model_class": FusionDenseNet,
        "output_dir": Path(
            "_validation_runs/results/densenet121_fusion"
        ),
        "checkpoint": Path(
            "_validation_runs/checkpoints/"
            "densenet121_fusion_best.pth"
        ),
    },

    "efficientnet_b3_fusion": {
        "model_class": FusionEffB3,
        "output_dir": Path(
            "_validation_runs/results/efficientnet_b3_fusion"
        ),
        "checkpoint": Path(
            "_validation_runs/checkpoints/"
            "efficientnet_b3_fusion_best.pth"
        ),
    },

    "mobilenetv3_large_fusion": {
        "model_class": FusionMobileNetV3,
        "output_dir": Path(
            "_validation_runs/results/mobilenetv3_large_fusion"
        ),
        "checkpoint": Path(
            "_validation_runs/checkpoints/"
            "mobilenetv3_large_fusion_best.pth"
        ),
    },
}

# ---------------------------------------------------------------------
# Train each fusion model
# ---------------------------------------------------------------------

for experiment_name, config in EXPERIMENTS.items():

    print(f"\n{'=' * 70}")
    print(f"Training: {experiment_name}")
    print(f"{'=' * 70}")

    output_dir = config["output_dir"]
    checkpoint_path = config["checkpoint"]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create fusion model
    fusion = config["model_class"](
        hf_dim=HF_DIM,
        num_classes=NUM_CLASSES,
        p_drop=0.2,
    ).to(device)

    # Freeze pretrained image backbone
    for p in fusion.backbone.parameters():
        p.requires_grad = False

    # Weighted cross-entropy
    criterion = create_criterion(
        Ytr,
        NUM_CLASSES,
        device,
    )

    # Optimizer parameter groups follow each fusion architecture
    if experiment_name == "densenet121_fusion":

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": fusion.head.parameters(),
                    "lr": LR_RUN,
                },
                {
                    "params": fusion.bn_img.parameters(),
                    "lr": LR_RUN,
                },
                {
                    "params": fusion.bn_hf.parameters(),
                    "lr": LR_RUN,
                },
            ],
            weight_decay=WEIGHT_DECAY,
        )

    elif experiment_name == "efficientnet_b3_fusion":

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": fusion.img_proj.parameters(),
                    "lr": LR_RUN,
                },
                {
                    "params": fusion.hf_proj.parameters(),
                    "lr": LR_RUN,
                },
                {
                    "params": fusion.head.parameters(),
                    "lr": LR_RUN,
                },
            ],
            weight_decay=WEIGHT_DECAY,
        )

    else:

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": fusion.img_proj.parameters(),
                    "lr": LR_RUN,
                },
                {
                    "params": fusion.hf_proj.parameters(),
                    "lr": LR_RUN,
                },
                {
                    "params": fusion.head.parameters(),
                    "lr": LR_RUN,
                },
            ],
            weight_decay=WEIGHT_DECAY,
        )

    # Reduce LR when validation macro-F1 plateaus
    scheduler = create_scheduler(
        optimizer
    )

    # Automatic mixed precision
    scaler, amp_ctx = create_amp()

    # Early-stopping state
    best_score = -1.0
    best_state = None
    best_epoch = -1
    best_val = None
    best_train = None

    pat = PATIENCE_RUN
    history = []

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------

    for ep in range(
        1,
        EPOCHS_RUN + 1,
    ):

        trM = epoch_fusion(
            model=fusion,
            X=Xtr,
            HF=HFtr_t,
            Y=Ytr,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp_ctx=amp_ctx,
            device=device,
            batch_size=BATCH_SIZE,
            train=True,
        )

        vaM = epoch_fusion(
            model=fusion,
            X=Xva,
            HF=HFva_t,
            Y=Yva,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp_ctx=amp_ctx,
            device=device,
            batch_size=BATCH_SIZE,
            train=False,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": ep,
                "lr": current_lr,

                "tr_loss": trM["loss"],
                "tr_acc": trM["acc"],
                "tr_prec": trM["prec"],
                "tr_rec": trM["rec"],
                "tr_f1": trM["f1"],
                "tr_auc": trM["auc"],
                "tr_mcc": trM["mcc"],

                "va_loss": vaM["loss"],
                "va_acc": vaM["acc"],
                "va_prec": vaM["prec"],
                "va_rec": vaM["rec"],
                "va_f1": vaM["f1"],
                "va_auc": vaM["auc"],
                "va_mcc": vaM["mcc"],
            }
        )

        print(
            f"[Ep {ep:02d}] "
            f"LR={current_lr:.2e} | "
            f"TR L={trM['loss']:.3f} "
            f"A={trM['acc']:.3f} "
            f"F1={trM['f1']:.3f} "
            f"AUC={trM['auc']:.3f} | "
            f"VA L={vaM['loss']:.3f} "
            f"A={vaM['acc']:.3f} "
            f"F1={vaM['f1']:.3f} "
            f"AUC={vaM['auc']:.3f}"
        )

        scheduler.step(
            vaM[STOP_METRIC]
        )

        score = vaM[
            STOP_METRIC
        ]

        if score > best_score:

            best_score = score

            best_state = {
                k: v.detach().cpu()
                for k, v
                in fusion.state_dict().items()
            }

            best_epoch = ep
            best_val = vaM.copy()
            best_train = trM.copy()

            pat = PATIENCE_RUN

        else:

            pat -= 1

            if pat <= 0:
                print("Early stop.")
                break


    # -----------------------------------------------------------------
    # Restore and save best checkpoint
    # -----------------------------------------------------------------

    if best_state is not None:

        torch.save(
            best_state,
            checkpoint_path,
        )

        fusion.load_state_dict(
            best_state,
            strict=True,
        )


    # -----------------------------------------------------------------
    # Save training history
    # -----------------------------------------------------------------

    log_df = pd.DataFrame(
        history
    )

    csv_path = (
        output_dir
        / "train_val_metrics.csv"
    )

    try:

        log_df.to_csv(
            csv_path,
            index=False,
        )

    except PermissionError:

        csv_path = output_dir / (
            "train_val_metrics_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".csv"
        )

        log_df.to_csv(
            csv_path,
            index=False,
        )


    # -----------------------------------------------------------------
    # Report best train and validation performance
    # -----------------------------------------------------------------

    print(
        f"\n=== Best {experiment_name} checkpoint "
        f"by val {STOP_METRIC.upper()} "
        f"at epoch {best_epoch} ==="
    )

    print("VAL:")

    print(
        f"  loss={best_val['loss']:.3f} | "
        f"acc={best_val['acc']:.3f} | "
        f"prec={best_val['prec']:.3f} | "
        f"rec={best_val['rec']:.3f} | "
        f"f1={best_val['f1']:.3f} | "
        f"auc={best_val['auc']:.3f} | "
        f"mcc={best_val['mcc']:.3f}"
    )

    print("TRAIN:")

    print(
        f"  loss={best_train['loss']:.3f} | "
        f"acc={best_train['acc']:.3f} | "
        f"prec={best_train['prec']:.3f} | "
        f"rec={best_train['rec']:.3f} | "
        f"f1={best_train['f1']:.3f} | "
        f"auc={best_train['auc']:.3f} | "
        f"mcc={best_train['mcc']:.3f}"
    )

    print(
        f"\nCheckpoint saved to: "
        f"{checkpoint_path}"
    )

    print(
        f"History CSV saved to: "
        f"{csv_path}"
    )


    # -----------------------------------------------------------------
    # Validation-set threshold tuning
    # -----------------------------------------------------------------

    va_prob = fusion_validation_probabilities(
        model=fusion,
        X=Xva,
        HF=HFva_t,
        amp_ctx=amp_ctx,
        batch_size=BATCH_SIZE,
    )

    yva_np = (
        Yva.detach()
        .cpu()
        .numpy()
    )

    threshold, best_f1_tune = (
        tune_threshold(
            va_prob,
            yva_np,
        )
    )

    print(
        f"[Threshold] Selected "
        f"τ*={threshold:.3f} "
        f"using VAL macro-F1="
        f"{best_f1_tune:.3f}"
    )


    # Save selected threshold for test evaluation
    threshold_path = (
        output_dir
        / "threshold.txt"
    )

    with open(
        threshold_path,
        "w",
    ) as f:

        f.write(
            f"{threshold:.6f}\n"
        )