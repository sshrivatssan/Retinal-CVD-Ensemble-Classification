from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data import (
    load_split_tensors,
    prepare_handcrafted_features,
)
from src.models import (
    build_baseline_model,
    FusionDenseNet,
    FusionEffB3,
    FusionMobileNetV3,
)
from src.training import (
    create_amp,
    create_criterion,
    metrics_from_logits,
)


# ---------------------------------------------------------------------
# Device and data settings
# ---------------------------------------------------------------------

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


SPLIT_DIR = Path(
    "_validation_runs/CVD_split"
)

HAND_FEATS = Path(
    r"D:\Research\CVD\CVD_Project\features_lbp_glcm_all.csv"
)

NUM_CLASSES = 2
BATCH_SIZE = 32

CLASS_NAMES = [
    "Healthy",
    "CVD",
]


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
# Load test data
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


tr_df = pd.read_csv(
    SPLIT_DIR / "train.csv"
)

va_df = pd.read_csv(
    SPLIT_DIR / "val.csv"
)

te_df = pd.read_csv(
    SPLIT_DIR / "test.csv"
)


# ---------------------------------------------------------------------
# Prepare handcrafted features for fusion models
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


# Automatic mixed precision
scaler, amp_ctx = create_amp()


# ---------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_baseline(
    model,
    X,
    Y,
    criterion,
    batch_size,
):
    """Evaluate a baseline CNN on the test set."""

    model.eval()

    N = X.size(0)

    total_loss = 0.0
    logits_all = []

    for i in range(
        0,
        N,
        batch_size,
    ):

        xb = X[i:i + batch_size]
        yb = Y[i:i + batch_size]

        with amp_ctx():
            out = model(xb)
            loss = criterion(out, yb)

        total_loss += (
            loss.item() * yb.size(0)
        )

        logits_all.append(
            out.detach()
        )

    logits = torch.cat(
        logits_all,
        dim=0,
    )

    y_true = (
        Y.detach()
        .cpu()
        .numpy()
    )

    mets = metrics_from_logits(
        logits,
        y_true,
    )

    mets["loss"] = (
        total_loss / N
    )

    y_pred = (
        logits.argmax(1)
        .detach()
        .cpu()
        .numpy()
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    return (
        mets,
        cm,
        y_pred,
    )


# ---------------------------------------------------------------------
# Fusion evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_fusion(
    model,
    X,
    HF,
    Y,
    threshold,
    batch_size,
):
    """Evaluate a fusion model using its validation-selected threshold."""

    model.eval()

    probs = []
    logits_all = []

    for i in range(
        0,
        X.size(0),
        batch_size,
    ):

        xb = X[i:i + batch_size]
        hb = HF[i:i + batch_size]

        with amp_ctx():
            logits = model(
                xb,
                hb,
            )

        probs.append(
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
            .detach()
            .cpu()
        )

        logits_all.append(
            logits.detach().cpu()
        )

    te_prob = torch.cat(
        probs
    ).numpy()

    te_logits = torch.cat(
        logits_all
    )

    y_true = (
    Y.detach()
    .cpu()
    .numpy()
    )

    y_pred = (
        te_prob >= threshold
    ).astype(int)

    acc = accuracy_score(
        y_true,
        y_pred,
    )

    prec, rec, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    auc = roc_auc_score(
        y_true,
        te_prob,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    mets = {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": auc,
        "mcc": mcc,
    }

    return (
        mets,
        cm,
        y_pred,
        te_prob,
        te_logits,
    )


# ---------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------

def plot_confusion_matrix(
    cm,
    title,
    output_path,
):
    """Plot and save a confusion matrix."""

    plt.figure(
        figsize=(6, 6)
    )

    im = plt.imshow(
        cm,
        cmap="Blues",
        interpolation="nearest",
    )

    plt.colorbar(
        im,
        fraction=0.046,
        pad=0.04,
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
    )

    plt.yticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
    )

    for i in range(
        cm.shape[0]
    ):
        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color="black",
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Baseline model configurations
# ---------------------------------------------------------------------

BASELINES = {
    "densenet121": {
        "checkpoint": Path(
            "_validation_runs/checkpoints/densenet121_best.pth"
        ),
        "result_dir": Path(
            "_validation_runs/results/densenet121"
        ),
    },

    "efficientnet_b3": {
        "checkpoint": Path(
            "_validation_runs/checkpoints/efficientnet_b3_best.pth"
        ),
        "result_dir": Path(
            "_validation_runs/results/efficientnet_b3"
        ),
    },

    "mobilenetv3_large": {
        "checkpoint": Path(
            "_validation_runs/checkpoints/mobilenetv3_large_best.pth"
        ),
        "result_dir": Path(
            "_validation_runs/results/mobilenetv3_large"
        ),
    },
}
# ---------------------------------------------------------------------
# Evaluate baseline models
# ---------------------------------------------------------------------

for model_name, config in BASELINES.items():

    print(f"\n{'=' * 70}")
    print(f"Evaluating baseline: {model_name}")
    print(f"{'=' * 70}")

    result_dir = config[
        "result_dir"
    ]

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = build_baseline_model(
        model_name,
        num_classes=NUM_CLASSES,
    ).to(device)

    state = torch.load(
        config["checkpoint"],
        map_location=device,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    # Weighted cross-entropy
    criterion = create_criterion(
        Ytr,
        NUM_CLASSES,
        device,
    )

    teM, cm, y_pred = (
        evaluate_baseline(
            model=model,
            X=Xte,
            Y=Yte,
            criterion=criterion,
            batch_size=BATCH_SIZE,
        )
    )

    print(
        "\n=== TEST METRICS ==="
    )

    print(
        f"loss={teM['loss']:.3f} | "
        f"acc={teM['acc']:.3f} | "
        f"prec={teM['prec']:.3f} | "
        f"rec={teM['rec']:.3f} | "
        f"f1={teM['f1']:.3f} | "
        f"auc={teM['auc']:.3f} | "
        f"mcc={teM['mcc']:.3f}"
    )

    print(
        "\nPer-class classification report:"
    )

    report = classification_report(
        Yte.detach().cpu().numpy(),
        y_pred,
        digits=3,
        zero_division=0,
    )

    print(report)

    print(
        "\nConfusion Matrix "
        "(rows=true, cols=pred):"
    )

    print(cm)

    # Save metrics
    pd.DataFrame(
        [
            {
                "loss": teM["loss"],
                "acc": teM["acc"],
                "prec": teM["prec"],
                "rec": teM["rec"],
                "f1": teM["f1"],
                "auc": teM["auc"],
                "mcc": teM["mcc"],
            }
        ]
    ).to_csv(
        result_dir / "test_metrics.csv",
        index=False,
    )

    np.savetxt(
        result_dir
        / "confusion_matrix_test.csv",
        cm.astype(int),
        fmt="%d",
        delimiter=",",
    )

    with open(
        result_dir
        / "classification_report_test.txt",
        "w",
    ) as f:
        f.write(report)

    plot_confusion_matrix(
        cm=cm,
        title=(
            f"{model_name} Confusion Matrix (Test)"
        ),
        output_path=(
            result_dir
            / "confusion_matrix_test.png"
        ),
    )


# ---------------------------------------------------------------------
# Fusion model configurations
# ---------------------------------------------------------------------

FUSION_MODELS = {
    "densenet121_fusion": {
        "model_class": FusionDenseNet,
        "checkpoint": Path(
            "_validation_runs/checkpoints/densenet121_fusion_best.pth"
        ),
        "result_dir": Path(
            "_validation_runs/results/densenet121_fusion"
        ),
    },

    "efficientnet_b3_fusion": {
        "model_class": FusionEffB3,
        "checkpoint": Path(
            "_validation_runs/checkpoints/efficientnet_b3_fusion_best.pth"
        ),
        "result_dir": Path(
            "_validation_runs/results/efficientnet_b3_fusion"
        ),
    },

    "mobilenetv3_large_fusion": {
        "model_class": FusionMobileNetV3,
        "checkpoint": Path(
            "_validation_runs/checkpoints/mobilenetv3_large_fusion_best.pth"
        ),
        "result_dir": Path(
            "_validation_runs/results/mobilenetv3_large_fusion"
        ),
    },
}

# ---------------------------------------------------------------------
# Evaluate fusion models
# ---------------------------------------------------------------------

for experiment_name, config in FUSION_MODELS.items():

    print(f"\n{'=' * 70}")
    print(f"Evaluating fusion: {experiment_name}")
    print(f"{'=' * 70}")

    result_dir = config[
        "result_dir"
    ]

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load validation-selected threshold
    threshold_path = (
        result_dir
        / "threshold.txt"
    )

    with open(
        threshold_path,
        "r",
    ) as f:
        threshold = float(
            f.read().strip()
        )

    # Build fusion model
    model = config[
        "model_class"
    ](
        hf_dim=HF_DIM,
        num_classes=NUM_CLASSES,
        p_drop=0.2,
    ).to(device)

    state = torch.load(
        config["checkpoint"],
        map_location=device,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    teM, cm, y_pred, te_prob, te_logits = (
    evaluate_fusion(
        model=model,
        X=Xte,
        HF=HFte_t,
        Y=Yte,
        threshold=threshold,
        batch_size=BATCH_SIZE,
    )
)

    print(
        f"\n=== TEST METRICS @ "
        f"τ*={threshold:.3f} ==="
    )

    print(
        f"acc={teM['acc']:.3f} | "
        f"prec={teM['prec']:.3f} | "
        f"rec={teM['rec']:.3f} | "
        f"f1={teM['f1']:.3f} | "
        f"auc={teM['auc']:.3f} | "
        f"mcc={teM['mcc']:.3f}"
    )

    print(
        "\nPer-class TEST report:"
    )

    report = classification_report(
        Yte.detach().cpu().numpy(),
        y_pred,
        digits=3,
        zero_division=0,
    )

    print(report)

    print(
        "\nConfusion Matrix "
        "(rows=true, cols=pred):"
    )

    print(cm)

    # Save test metrics
    pd.DataFrame(
        [
            {
                "threshold": threshold,
                "acc": teM["acc"],
                "prec": teM["prec"],
                "rec": teM["rec"],
                "f1": teM["f1"],
                "auc": teM["auc"],
                "mcc": teM["mcc"],
            }
        ]
    ).to_csv(
        result_dir
        / "test_metrics.csv",
        index=False,
    )

    np.savetxt(
        result_dir
        / "confusion_matrix_test.csv",
        cm.astype(int),
        fmt="%d",
        delimiter=",",
    )

    with open(
        result_dir
        / "classification_report_test.txt",
        "w",
    ) as f:
        f.write(report)

    plot_confusion_matrix(
        cm=cm,
        title=(
            f"{experiment_name} "
            f"Confusion Matrix (Test @ τ*)"
        ),
        output_path=(
            result_dir
            / "confusion_matrix_test.png"
        ),
    )