import numpy as np
import torch


@torch.no_grad()
def mixval(model, x_target, logits, lam: float = 0.55, batch_size: int = 128, seed: int = 0,
           device: str = "cpu") -> float:
    """
    MixVal ICE-inter score (Hu et al., NeurIPS 2023; official val_mixval_inter.py), computed on the INPUTS:

      1. pseudo labels = argmax of the model's predictions on the clean target samples (`logits`);
      2. the target set is put in a fixed shuffled order (the official code reads shuffled target lists, one per run;
         `seed` plays that role, seed=None keeps the given order) and split into batches of `batch_size`;
         within each batch sample i is paired with sample B-1-i (reversed batch);
      3. each pair is mixed in input space, x = lam * x_a + (1 - lam) * x_b, and the model is run on the mixture;
      4. score = accuracy of the mixed predictions w.r.t. the dominant pseudo label (that of x_a for lam > 0.5),
         over the pairs whose two pseudo labels differ (inter-class pairs).

    model:     the trained model (e.g. restored with utils.checkpoint.load_weights); must provide predict(x).
    x_target:  target inputs, a tensor (N, ...) or an indexable dataset returning (x, ...) per sample,
               in the same order as `logits`.
    logits:    (N, C) predictions of the same model on x_target (e.g. checkpoint['target_test']['logits']).
    Higher is better. Returns NaN if no inter-class pair exists (all samples share one pseudo label).
    """
    logits = torch.as_tensor(np.asarray(logits)).float()
    N = logits.shape[0]
    pl = logits.argmax(dim=1)
    order = np.arange(N) if seed is None else np.random.RandomState(seed).permutation(N)

    def get(idx):
        if torch.is_tensor(x_target):
            return x_target[torch.as_tensor(idx)]
        return torch.stack([torch.as_tensor(x_target[int(i)][0] if isinstance(x_target[int(i)], (tuple, list))
                                            else x_target[int(i)]) for i in idx])

    was_training = model.training
    model.eval()
    correct = total = 0
    for start in range(0, N, batch_size):
        idx = order[start:start + batch_size]
        rev = idx[::-1].copy()
        diff = (pl[idx] != pl[rev]).numpy()
        if not diff.any():
            continue
        x_a, x_b = get(idx[diff]), get(rev[diff])
        mix = (lam * x_a + (1 - lam) * x_b).float().to(device)
        pred = model.predict(mix).argmax(dim=1).cpu()
        dominant = pl[idx[diff]] if lam >= 0.5 else pl[rev[diff]]
        correct += int((pred == dominant).sum())
        total += int(diff.sum())
    model.train(was_training)
    return correct / total if total else float("nan")


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "checkpoint", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "checkpoint.py"))
    checkpoint = importlib.util.module_from_spec(spec); spec.loader.exec_module(checkpoint)

    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")
    model = build_model()                       # the same algorithm/backbone configuration used for training
    checkpoint.load_weights(model, ckpt)
    x_target = target_test_dataset              # same order as the saved target_test predictions

    score = mixval(model, x_target, ckpt["target_test"]["logits"])
    print(f"MixVal (ICE-inter) Score: {score:.6f}  (higher is better)")
