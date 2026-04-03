import numpy as np
from sklearn.metrics import balanced_accuracy_score

def compute_accuracy(logits: np.ndarray, gts: np.ndarray) -> float:
    logits = np.asarray(logits)
    gts = np.asarray(gts).astype(int)
    
    preds = np.argmax(logits, axis=1)

    acc = balanced_accuracy_score(gts, preds)

    return float(acc)
    
EPS = 1e-12
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits) / temperature
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=1, keepdims=True) + EPS)
