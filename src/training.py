from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)


# Training settings
LR_RUN = 1e-4
EPOCHS_RUN = 40
PATIENCE_RUN = 8
WEIGHT_DECAY = 1e-4
STOP_METRIC = "f1"
AUG_P = 0.5


def metrics_from_logits(logits_cat, labels_np):
    """Compute classification metrics from model logits."""

    y_true = labels_np

    # Positive-class probabilities
    y_prob = (
        torch.softmax(logits_cat, dim=1)[:, 1]
        .detach()
        .cpu()
        .numpy()
    )

    # Default decision threshold
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)

    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(
        y_true,
        y_pred,
    ).tolist()

    return {
        "acc": acc,
        "prec": pr,
        "rec": rc,
        "f1": f1,
        "auc": auc,
        "mcc": mcc,
        "cm": cm,
    }


def compute_class_weights(Ytr, num_classes, device):
    """Compute class weights for weighted cross-entropy."""

    class_counts = torch.bincount(
        Ytr.detach().cpu(),
        minlength=num_classes,
    ).float()

    class_weights = (
        class_counts.sum()
        / (num_classes * class_counts)
    )

    class_weights = class_weights.to(device)

    return class_counts, class_weights


def create_criterion(Ytr, num_classes, device):
    """Create weighted cross-entropy loss."""

    class_counts, class_weights = compute_class_weights(
        Ytr,
        num_classes,
        device,
    )

    print("Class counts:", class_counts.tolist())
    print(
        "Class weights:",
        class_weights.detach().cpu().tolist(),
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    return criterion


def create_optimizer(parameters):
    """Create AdamW optimizer."""

    return torch.optim.AdamW(
        parameters,
        lr=LR_RUN,
        weight_decay=WEIGHT_DECAY,
    )


def create_scheduler(optimizer):
    """Create validation macro-F1 learning-rate scheduler."""

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )


def create_amp():
    """Create AMP scaler and autocast context."""

    device_is_cuda = torch.cuda.is_available()

    scaler = (
        torch.amp.GradScaler("cuda")
        if device_is_cuda
        else None
    )

    def amp_ctx():
        return (
            torch.amp.autocast("cuda")
            if device_is_cuda
            else nullcontext()
        )

    return scaler, amp_ctx


def augment_gpu(x, model):
    """Apply training-time image augmentation."""

    if not model.training:
        return x

    # Random horizontal flip
    if torch.rand(1, device=x.device) < AUG_P:
        x = torch.flip(x, dims=[3])

    # Brightness variation
    b = (
        torch.rand(
            x.size(0),
            1,
            1,
            1,
            device=x.device,
        ) - 0.5
    ) * 0.10

    # Contrast variation
    c = 1.0 + (
        torch.rand(
            x.size(0),
            1,
            1,
            1,
            device=x.device,
        ) - 0.5
    ) * 0.10

    x = (x * c + b).clamp_(-4, 4)

    return x


def epoch_baseline(
    model,
    X,
    Y,
    criterion,
    optimizer,
    scaler,
    amp_ctx,
    device,
    batch_size,
    train=True,
):
    """Run one training or validation epoch for a baseline model."""

    model.train(train)

    N = X.size(0)

    order = (
        torch.randperm(N, device=device)
        if train
        else torch.arange(N, device=device)
    )

    total_loss = 0.0
    logits_all = []

    for i in range(0, N, batch_size):

        idx = order[i:i + batch_size]

        xb = X[idx]
        yb = Y[idx]

        if train:
            xb = augment_gpu(
                xb,
                model,
            )

        with amp_ctx():
            out = model(xb)
            loss = criterion(out, yb)

        if train:

            optimizer.zero_grad(
                set_to_none=True
            )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                optimizer.step()

        total_loss += (
            loss.item() * yb.size(0)
        )

        logits_all.append(
            out.detach()
        )

    logits_cat = torch.cat(
        logits_all,
        dim=0,
    )

    mets = metrics_from_logits(
        logits_cat,
        Y.detach().cpu().numpy(),
    )

    mets["loss"] = total_loss / N

    return mets


def epoch_fusion(
    model,
    X,
    HF,
    Y,
    criterion,
    optimizer,
    scaler,
    amp_ctx,
    device,
    batch_size,
    train=True,
):
    """Run one training or validation epoch for a fusion model."""

    model.train(train)

    N = X.size(0)

    order = (
        torch.randperm(N, device=device)
        if train
        else torch.arange(N, device=device)
    )

    total_loss = 0.0
    logits_all = []

    for i in range(0, N, batch_size):

        idx = order[i:i + batch_size]

        xb = X[idx]
        hb = HF[idx]
        yb = Y[idx]

        if train:
            xb = augment_gpu(
                xb,
                model,
            )

        with amp_ctx():
            out = model(xb, hb)
            loss = criterion(out, yb)

        if train:

            optimizer.zero_grad(
                set_to_none=True
            )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                optimizer.step()

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

    mets = metrics_from_logits(
        logits,
        Y.detach().cpu().numpy(),
    )

    mets["loss"] = total_loss / N

    return mets


@torch.no_grad()
def fusion_validation_probabilities(
    model,
    X,
    HF,
    amp_ctx,
    batch_size,
):
    """Compute positive-class probabilities on the validation set."""

    model.eval()

    probs = []

    for i in range(
        0,
        X.size(0),
        batch_size,
    ):

        xb = X[i:i + batch_size]
        hb = HF[i:i + batch_size]

        with amp_ctx():
            logits = model(xb, hb)

        probs.append(
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
            .detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        probs,
        axis=0,
    )


def tune_threshold(va_prob, yva_np):
    """Select the threshold maximizing validation macro-F1."""

    ths = np.linspace(
        0.2,
        0.8,
        61,
    )

    best_thr = 0.5
    best_f1_tune = -1.0

    for t in ths:

        pred = (
            va_prob >= t
        ).astype(int)

        _, _, f1_t, _ = (
            precision_recall_fscore_support(
                yva_np,
                pred,
                average="macro",
                zero_division=0,
            )
        )

        if f1_t > best_f1_tune:
            best_f1_tune = f1_t
            best_thr = t

    return (
        float(best_thr),
        float(best_f1_tune),
    )