"""Linear probes on hidden states: does layer l linearly encode 'this answer will be correct'?"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def layerwise_auroc(X, y, C: float = 0.01, folds: int = 5, seed: int = 0):
    """X: (N, L, d), y: (N,) 1=correct. Returns cross-validated AUROC per layer
    for predicting correctness from the hidden state."""
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    out = []
    for l in range(X.shape[1]):
        p = cross_val_predict(_clf(C), X[:, l].astype(np.float32), y, cv=skf, method="predict_proba")[:, 1]
        out.append(roc_auc_score(y, p))
    return np.array(out)


def probe_direction(X_layer, y, C: float = 0.01):
    """Fit a probe on one layer; return the unit 'correctness direction' in raw
    activation space (for steering / patching experiments)."""
    clf = _clf(C).fit(X_layer.astype(np.float32), np.asarray(y).astype(int))
    scaler, lr = clf.steps[0][1], clf.steps[1][1]
    w = lr.coef_[0] / scaler.scale_
    return w / np.linalg.norm(w)
