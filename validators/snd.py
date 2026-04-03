import numpy as np
import torch

EPS = 1e-12

from utils import softmax

def snd(logits: np.ndarray, tau: float = 0.05) -> float:
    logits = np.asarray(logits)
    probs = softmax(logits)
    N = probs.shape[0]

    if N <= 1:
        return float("nan")

    # L2 Normalise features
    norms = np.linalg.norm(probs, axis=1, keepdims=True)
    feats = probs / (norms + EPS)

    # Pairwise cosine similarity matrix, temperature-scaled
    sim = (feats @ feats.T) / tau

    # Mask self-similarity with -1e9 (np.fill_diagonal modifies in-place)
    np.fill_diagonal(sim, -1e9)

    # Convert to soft neighbourhood distribution
    sim_probs = softmax_np(sim)

    # Per-sample entropy, then mean
    H = -(sim_probs * np.log(sim_probs + 1e-8)).sum(axis=1)
    return float(H.mean())


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")
    
    logits = ckpt["target_test"]["logits"]
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()
    
    score = snd(logit)
    print(f"SND Score: {score:.6f}  (lower is better)")

