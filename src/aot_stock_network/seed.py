"""Global reproducibility — deterministic random seeding.

Call :func:`set_seed()` once at application startup to lock all
random-number generators to a known state.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

from aot_stock_network.config import settings


def set_seed(seed: Optional[int] = None) -> None:
    """Lock random generators for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed (default: ``settings.random_state``).
    """
    seed = seed if seed is not None else settings.random_state
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
