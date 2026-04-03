import numpy as np
import torch

EPS = 1e-12

from utils import softmax

def entropy(logits: np.ndarray) -> float:
    logits = np.asarray(logits)
    p = softmax(logits)
    return float((-(p * np.log(p + EPS)).sum(axis=1)).mean())


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")

    logits = ckpt["target_test"]["logits"]
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()

    score = entropy(logits)
    print(f"Entropy Score: {score:.6f}  (lower is better)")

