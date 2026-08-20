import random

import numpy as np
import torch


SEED = 42


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch():
    """Configure PyTorch execution settings."""

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def get_device():
    """Select the CUDA device and report hardware information."""

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

    return device