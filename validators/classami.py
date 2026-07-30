import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

def classami(logits: np.ndarray, features: np.ndarray, random_state: int = 42) -> float:
    logits = np.asarray(logits)
    features = np.asarray(features)
    num_classes = logits.shape[1]

    pred_labels = np.argmax(logits, axis=1)

    # Cluster the feature space into num_classes clusters
    kmeans = KMeans(n_clusters=num_classes, n_init="auto", random_state=random_state)
    cluster_labels = kmeans.fit_predict(features)

    # Adjusted Mutual Information between predicted classes and feature clusters
    return float(adjusted_mutual_info_score(pred_labels, cluster_labels))


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")

    logits = ckpt["target_test"]["logits"]
    features = ckpt["target_test"]["features"]
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    score = classami(logits, features)
    print(f"ClassAMI Score: {score:.6f}  (higher is better)")
