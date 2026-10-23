"""Deterministic seeding for python, numpy and (if installed) torch."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every random number generator we use.

    Args:
        seed: the seed value (configs/default.yaml -> seed).
        deterministic: also ask torch/cuDNN for deterministic kernels (slightly slower on GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; seeded python and numpy only")
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
