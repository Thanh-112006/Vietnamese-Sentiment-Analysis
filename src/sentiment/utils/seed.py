import random

import numpy as np


def set_seed(seed: int) -> None:
    """Đặt seed cho Python, NumPy và PyTorch nếu đã cài."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return

    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
