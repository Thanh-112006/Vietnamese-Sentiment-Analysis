from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_fasttext(
    path: Path, vocabulary: Mapping[str, int], *, embedding_dim: int = 300
) -> NDArray[np.float32]:
    """Nạp vector fastText vào ma trận theo chỉ số của từ vựng."""
    raise NotImplementedError
