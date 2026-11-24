"""Calibration and error-detection metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def ece(conf, correct, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def auroc_error_detection(uncertainty, correct) -> float:
    """AUROC of an uncertainty score (higher = less trustworthy) at detecting wrong answers."""
    correct = np.asarray(correct).astype(int)
    return float(roc_auc_score(1 - correct, np.asarray(uncertainty, float)))
