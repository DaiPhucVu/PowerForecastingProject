"""Project settings shared by every script. Import from here, don't copy the values."""

import os
import random

import numpy as np

SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and TensorFlow so runs can be repeated."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(seed)
