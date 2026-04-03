import numpy as np
import torch

EPS = 1e-12

from utils import softmax

def mcc(logits: np.ndarray, temperature: float = 2.5) -> float:
    logits = np.asarray(logits)
    batch_size, num_classes = logits.shape

    predictions = softmax(logits, temperature=temperature)

    # Entropy-based sample re-weighting
    entropy_val = -(predictions * np.log(predictions + 1e-5)).sum(axis=1)
    entropy_weight = 1 + np.exp(-entropy_val)
    
    # Normalize to batch size and reshape to (N, 1)
    entropy_weight = (batch_size * entropy_weight / np.sum(entropy_weight))[:, np.newaxis]

    # Weighted class confusion matrix: (C, N) @ (N, C) -> (C, C)
    ccm = (predictions * entropy_weight).T @ predictions
    
    # Row-normalise
    ccm = ccm / (np.sum(ccm, axis=1, keepdims=True) + EPS)

    # Calculate loss: (sum of all elements - sum of diagonal elements) / C
    mcc_loss = (np.sum(ccm) - np.trace(ccm)) / num_classes
    return float(mcc_loss)


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")
    
   logits = ckpt["target_test"]["logits"]
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()
    
    score = mcc(logits)
    print(f"MCC Score: {score:.6f}  (lower is better)")

