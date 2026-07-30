import numpy as np
import torch
import torch.nn.functional as F

def mixval(logits: np.ndarray, lam: float = 0.55) -> float:
    """
    MixVal ICE-inter score: accuracy of model predictions on inter-class mixed
    pairs, computed via logit-space linear interpolation. Fully unsupervised —
    uses pseudo-labels from argmax(logits) only.
    """
    logits = np.asarray(logits)
    N, C = logits.shape

    logits_t = torch.from_numpy(logits.copy()).float()
    pl = F.one_hot(logits_t.argmax(dim=-1), num_classes=C).float()
    rand_idx = torch.arange(N).flip([0])

    pl_a, pl_b = pl, pl[rand_idx]
    mix_logits = lam * logits_t + (1 - lam) * logits_t[rand_idx]
    mix_labels = lam * pl_a + (1 - lam) * pl_b
    diff_mask = pl_a.argmax(dim=-1) != pl_b.argmax(dim=-1)

    if diff_mask.sum() == 0:
        return float("nan")

    correct = (mix_logits[diff_mask].argmax(dim=-1) == mix_labels[diff_mask].argmax(dim=-1))
    return float(correct.float().mean().item())


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")

    logits = ckpt["target_test"]["logits"]
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()

    score = mixval(logits)
    print(f"MixVal (ICE-inter) Score: {score:.6f}  (higher is better)")
