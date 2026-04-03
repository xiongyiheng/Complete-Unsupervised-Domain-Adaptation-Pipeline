import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple

EPS = 1e-12

def get_weight(source_feature: np.ndarray, target_feature: np.ndarray, validation_feature: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Trains a domain classifier to compute importance weights."""
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape

    all_feature = np.concatenate((source_feature, target_feature), axis=0)
    all_label = np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        all_feature, all_label, train_size=0.8, random_state=random_state, stratify=all_label
    )

    base = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            solver="saga", penalty="l2", C=1.0 / 1e-3,
            max_iter=50, n_jobs=-1, verbose=0, random_state=0
        )
    )
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    P = clf.predict_proba(validation_feature)
    ratio = (P[:, 1:2] + 1e-8) / (P[:, 0:1] + 1e-8)
    w = (1.0 / ratio) * (N_t / max(N_s, 1))
    return w

def get_dev_risk(weight: np.ndarray, error: np.ndarray) -> float:
    """Computes the Deep Embedded Validation (DEV) Risk."""
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1) + EPS
    eta = - cov / var_w
    
    dev_score = np.mean(weighted_error) + eta * np.mean(weight) - eta
    return float(dev_score)

def get_devn_weights(weight: np.ndarray) -> np.ndarray:
    """Normalizes the weights for DEVN."""
    max_w = np.max(weight)
    if max_w < EPS:
        return weight

    V = weight / max_w
    V_mean = np.mean(V)
    W_norm = V - V_mean + 1.0
    return W_norm

def compute_dev(
    feats_source_train: np.ndarray, 
    feats_target_train: np.ndarray, 
    feats_val: np.ndarray, 
    logits_val: np.ndarray, 
    gts_val: np.ndarray
) -> Tuple[float, float]:
    """
    DEV & DEVN Score: Deep Embedded Validation Risk.
    Direction: MINIMIZE (lower risk = better).
    Returns: (dev_score, devn_score)
    """
    y_pred = np.argmax(logits_val, axis=1)
    err = (y_pred != gts_val.astype(int)).astype(np.float32).reshape(-1, 1)

    # 1. Get raw weights
    w = get_weight(feats_source_train, feats_target_train, feats_val)

    # 2. Get DEV Score
    dev_score = get_dev_risk(w, err)

    # 3. Get DEVN score (Normalized DEV)
    w_norm = get_devn_weights(w)
    devn_score = get_dev_risk(w_norm, err)

    return dev_score, devn_score


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")

    # 1. Load source train features
    f_src_tr = np.asarray(ckpt["source_train"]["features"])
    
    # 2. Load target train features
    f_tgt_tr = np.asarray(ckpt["target_train"]["features"])
    
    # 3. Load validation set (source test)
    f_val = np.asarray(ckpt["source_test"]["features"])
    l_val = np.asarray(ckpt["source_test"]["logits"])
    g_val = np.asarray(ckpt["source_test"]["gts"]).reshape(-1)

    dev, devn = compute_dev(f_src_tr, f_tgt_tr, f_val, l_val, g_val)
    print(f"DEV Risk Score:  {dev:.6f}  (lower is better)")
    print(f"DEVN Risk Score: {devn:.6f}  (lower is better)")

