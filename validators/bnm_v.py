import numpy as np
import torch
EPS = 1e-12

from utils import softmax

def bnm(logits: np.ndarray) -> float:
    logits = np.asarray(logits)
    probs = softmax(logits)

    # SVD on the probability matrix
    # full_matrices=False makes it equivalent to PyTorch's default reduced SVD
    _, singular_values, _ = np.linalg.svd(probs, full_matrices=False)

    # Nuclear norm = mean of singular values
    nuclear_norm = np.mean(singular_values)
    return float(nuclear_norm)


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")
    
    logits = ckpt["target_test"]["logits"]
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()
    
    score = bnm(logits)
    print(f"BNM Score: {score:.6f}  (higher is better)")
