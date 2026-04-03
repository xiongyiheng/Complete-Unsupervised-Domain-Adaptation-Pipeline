import numpy as np
import torch

EPS = 1e-12

from utils import softmax

def infomax(logits: np.ndarray) -> float:
    logits = np.asarray(logits)
    p = softmax(logits)

    # Marginal entropy: entropy of the mean prediction across all samples
    p_bar = p.mean(axis=0)
    H_marginal = -(p_bar * np.log(p_bar + EPS)).sum()

    # Conditional entropy: mean per-sample entropy
    H_conditional = (-(p * np.log(p + EPS)).sum(axis=1)).mean()

    return float(H_marginal - H_conditional)


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")

    logits = ckpt["target_test"]["logits"]
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()

    score = infomax(logits)
    print(f"InfoMax Score: {score:.6f}  (higher is better)")

