import math
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors

from utils import softmax


def uniformity(W: np.ndarray) -> float:
    """Official utils.uniformity: mean squared deviation of the pairwise angles between class weight vectors from
    the ideal angle arccos(-1/(C-1))."""
    Wt = torch.as_tensor(np.asarray(W), dtype=torch.float32)
    if Wt.dim() > 2:
        Wt = Wt.view(Wt.size(0), -1)
    n = Wt.shape[0]
    W_norm = F.normalize(Wt, p=2, dim=1)
    cosine = torch.matmul(W_norm, W_norm.t())
    cosine_off = cosine.flatten()[:-1].view(n - 1, n + 1)[:, 1:]
    theta = torch.acos(cosine_off.clamp(-1.0, 1.0))  # clamp only guards float round-off
    theta0 = torch.acos(torch.tensor(-1.0 / (n - 1)))
    return float((theta - theta0).pow(2).mean().item())


def hopkins(features: np.ndarray, rng: np.random.RandomState) -> float:
    """Hopkins statistic, 5% sample, k = 2 nearest neighbours (official get_transfer_score)."""
    X = np.asarray(features, dtype=np.float64)
    N, D = X.shape
    sample_size = max(1, int(N * 0.05))
    X_uniform = rng.uniform(X.min(axis=0), X.max(axis=0), (sample_size, D))
    X_sample = X[rng.choice(N, size=sample_size, replace=False)]
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    u_sum = float(np.sum(nbrs.kneighbors(X_uniform, n_neighbors=2)[0][:, 0]))
    w_sum = float(np.sum(nbrs.kneighbors(X_sample, n_neighbors=2)[0][:, 1]))
    return u_sum / (u_sum + w_sum)


def im_loss(logits: np.ndarray, batch_size: int, rng: np.random.RandomState) -> float:
    """Mean over shuffled mini-batches (drop_last) of mean entropy - marginal entropy, as the official loop over
    the target loader; batch_size <= 0 uses the whole split as one batch."""
    probs = softmax(np.asarray(logits, dtype=np.float64))
    N = probs.shape[0]
    if batch_size <= 0 or batch_size > N:
        batches = [np.arange(N)]
    else:
        perm = rng.permutation(N)
        batches = [perm[i:i + batch_size] for i in range(0, N - batch_size + 1, batch_size)]
    vals = []
    for idx in batches:
        p = probs[idx]
        ent = float(np.mean(-(p * np.log(p + 1e-5)).sum(axis=1)))
        m = p.mean(axis=0)
        vals.append(ent - float(-(m * np.log(m + 1e-6)).sum()))
    return float(np.mean(vals))


def transferscore(head_features, logits: np.ndarray, head_weights, seed: int = 42, batch_size: int = 32) -> float:
    """
    TransferScore (Yang et al., ICLR 2024; official train.py get_transfer_score):
        score = H - im_loss / log(C) - u
      H        Hopkins statistic of the features fed to the final classifier layer (official: bottleneck output f,
               with predictions = head(f));
      im_loss  batch-averaged mean entropy - marginal entropy (= -InfoMax);
      u        uniformity of the final classifier layer's weight matrix (official: classifier.head weight).
    head_features: (N, D) or (H, N, D) for models with H heads (checkpoint['target_train']['head_features']);
    head_weights:  list of the H final-layer weight matrices (checkpoint['head_weights']).
    For several heads, H and u are averaged over the heads. Higher is better.
    """
    hf = np.asarray(head_features, dtype=np.float64)
    hf = hf[None] if hf.ndim == 2 else hf
    logits = np.asarray(logits, dtype=np.float64)
    C = logits.shape[1]
    rng = np.random.RandomState(seed)
    H = float(np.mean([hopkins(f, rng) for f in hf]))
    ent = im_loss(logits, batch_size, rng)
    u = float(np.mean([uniformity(np.asarray(W)) for W in head_weights]))
    return float(H - ent / math.log(C) - u)


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")

    split = ckpt["target_train"]          # the official score uses the (unlabelled) target training data
    score = transferscore(split["head_features"].numpy(), split["logits"].numpy(),
                          [W.numpy() for W in ckpt["head_weights"]])
    print(f"TransferScore: {score:.6f}  (higher is better)")
