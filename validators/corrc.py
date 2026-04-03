import numpy as np
import torch

EPS = 1e-12

from utils import softmax

def corrc_prob(logits: np.ndarray) -> float:
    """
    Corr-C computed from logits (Probability mode).
    Corr-C = trace(P^T P) / ||P^T P||_F
    Direction: MAXIMIZE (higher is better).
    """
    logits = np.asarray(logits)
    P = softmax(logits)

    A = P.T @ P  # (C, C) Gram matrix over classes
    num = np.trace(A)
    den = np.linalg.norm(A, ord="fro") + EPS
    return float(num / den)

def _corrcoef_from_features(F: np.ndarray) -> np.ndarray:
    """Helper: Feature-feature correlation matrix (D x D)."""
    F = np.asarray(F, dtype=np.float64)
    mu = F.mean(axis=0, keepdims=True)
    sd = F.std(axis=0, ddof=1, keepdims=True)
    
    Z = (F - mu) / (sd + EPS)
    N = F.shape[0]
    
    C = (Z.T @ Z) / max(N - 1, 1)
    np.clip(C, -1.0, 1.0, out=C)
    return C

def corrc_features(f_source: np.ndarray, f_target: np.ndarray, norm: str = "sumnorm") -> float:
    """
    Corr-C computed from source and target features (Feature mode).
    Direction: MAXIMIZE (higher = more consistent correlation structure).
    """
    Cs = _corrcoef_from_features(f_source)
    Ct = _corrcoef_from_features(f_target)
    
    num = np.linalg.norm(Cs - Ct, ord="fro")
    
    if norm == "sumnorm":
        denom = np.linalg.norm(Cs, ord="fro") + np.linalg.norm(Ct, ord="fro") + EPS
    elif norm == "bydim":
        denom = float(f_source.shape[1])
    else:
        raise ValueError("norm must be 'sumnorm' or 'bydim'")
        
    return float(1.0 - num / denom)


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")

    # ==========================================
    # 1. Probability Mode (using target logits)
    # ==========================================
    logits = ckpt["target_test"]["logits"]
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()

    score_prob = corrc_prob(logits)
    print(f"Corr-C (Prob Mode): {score_prob:.6f}  (higher is better)")


    # ==========================================
    # 2. Features Mode (using source & target)
    # ==========================================
    # Note: fallback keys depending on how they are saved in your dict
    f_source = ckpt["source_test"].get("features", ckpt["source_test"].get("feats"))
    f_target = ckpt["target_test"].get("features", ckpt["target_test"].get("feats"))
    
    if isinstance(f_source, torch.Tensor):
        f_source = f_source.numpy()
    if isinstance(f_target, torch.Tensor):
        f_target = f_target.numpy()

    score_feat = corrc_features(f_source, f_target, norm="sumnorm")
    print(f"Corr-C (Feature Mode): {score_feat:.6f}  (higher is better)")

