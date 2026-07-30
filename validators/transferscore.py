import math
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors

from utils import softmax

EPS = 1e-12

def _uniformity(W: np.ndarray) -> float:
    """Mean squared angular deviation of class weight vectors from the ideal inter-class angle."""
    Wt = torch.from_numpy(W.astype(np.float32))
    n = Wt.shape[0]
    W_norm = F.normalize(Wt, p=2, dim=1)
    cosine = torch.matmul(W_norm, W_norm.t())
    cosine_off = cosine.flatten()[:-1].view(n - 1, n + 1)[:, 1:]
    cosine_off = cosine_off.clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cosine_off)
    theta0 = torch.acos(torch.tensor(-1.0 / (n - 1)).clamp(-1 + 1e-6, 1 - 1e-6))
    return float((theta - theta0).pow(2).mean().item())


def transferscore(features: np.ndarray, logits: np.ndarray, seed: int = 42) -> float:
    """
    TransferScore = H - entropy_loss / log(C) - u
      H            = Hopkins statistic on features (clustering tendency; higher is better)
      entropy_loss = mean per-sample entropy - marginal entropy (= -InfoMax)
      u            = classifier head weight uniformity, with W recovered via
                     least squares (features @ W.T ≈ logits), since head weights
                     are not stored in the checkpoint.
    """
    features = np.asarray(features, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    N, C = logits.shape
    D = features.shape[1]

    rng = np.random.RandomState(seed)

    # Hopkins statistic (5% subsampling, kNN k=2)
    sample_size = max(1, int(N * 0.05))
    X_uniform = rng.uniform(features.min(axis=0), features.max(axis=0), (sample_size, D))
    random_indices = rng.choice(N, size=sample_size, replace=False)
    X_sample = features[random_indices]

    nbrs = NearestNeighbors(n_neighbors=2).fit(features)
    u_distances = nbrs.kneighbors(X_uniform, n_neighbors=2)[0][:, 0]
    w_distances = nbrs.kneighbors(X_sample, n_neighbors=2)[0][:, 1]

    u_sum = float(np.sum(u_distances))
    w_sum = float(np.sum(w_distances))
    H = u_sum / (u_sum + w_sum) if (u_sum + w_sum) > EPS else 0.5

    # Entropy loss: mean per-sample entropy - marginal entropy (= -InfoMax)
    probs = softmax(logits)
    mean_entropy = float(np.mean(-(probs * np.log(probs + 1e-5)).sum(axis=1)))
    marginal = probs.mean(axis=0)
    marginal_entropy = float(-(marginal * np.log(marginal + 1e-6)).sum())
    entropy_loss = mean_entropy - marginal_entropy

    # Classifier head uniformity, with W recovered via least squares
    features_aug = np.hstack([features, np.ones((N, 1))])
    theta_lstsq, _, _, _ = np.linalg.lstsq(features_aug, logits, rcond=None)
    W_hat = theta_lstsq[:D].T
    u = _uniformity(W_hat)

    return float(H - entropy_loss / math.log(C) - u)


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")

    logits = ckpt["target_test"]["logits"]
    features = ckpt["target_test"]["features"]
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    score = transferscore(features, logits)
    print(f"TransferScore: {score:.6f}  (higher is better)")
