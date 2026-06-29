import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from config import RANDOM_STATE

import numpy as np


def set_seed(seed: int = RANDOM_STATE):
    np.random.seed(seed)


def clamp_predictions(preds: np.ndarray, lower: float = 0.0) -> np.ndarray:
    return np.maximum(preds, lower)
