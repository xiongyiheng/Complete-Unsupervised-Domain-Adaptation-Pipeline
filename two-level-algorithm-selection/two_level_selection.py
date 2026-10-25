"""
Two-level agreement selection (Xiong et al., MICCAI Workshop 2026).

Inputs (one row per candidate checkpoint, same target samples for every checkpoint):
    table   pandas DataFrame with a column 'algorithm' and one column per validator score
    probs   array (K, N, C): target predictions (probabilities or logits) of the K checkpoints, in table row order
    directions  {validator: 'max' | 'min'}  (whether a higher or a lower score is better)

Level 1  For every algorithm a and validator v, v nominates its best checkpoint of a:  theta_{a,v}.
Level 2  The nominated checkpoints of all algorithms vote per target sample (one vote per (a, v) nomination);
         the majority class is the agreement reference y~.
Select   Every checkpoint of the pool is scored by its class-balanced agreement with y~; the best one is deployed.
"""
import numpy as np
import pandas as pd


def nominate(table: pd.DataFrame, directions: dict) -> pd.DataFrame:
    """Level 1: one nomination per (algorithm, validator). Returns rows (algorithm, validator, row) where row is the
    position of the nominated checkpoint in `table`. Checkpoints without a score (NaN) are skipped; ties go to the
    first checkpoint in table order."""
    rows = []
    for algo, sub in table.groupby('algorithm', sort=True):
        for v, d in directions.items():
            s = sub[v].to_numpy(dtype=float)
            ok = np.isfinite(s)
            if not ok.any():
                continue
            s = np.where(ok, s if d == 'max' else -s, -np.inf)
            rows.append(dict(algorithm=algo, validator=v, row=int(sub.index[np.argmax(s)])))
    return pd.DataFrame(rows)


def agreement_reference(probs: np.ndarray, nominated_rows) -> np.ndarray:
    """Level 2: per-sample majority vote over the predicted classes of the nominated checkpoints (a checkpoint
    nominated by several (algorithm, validator) pairs votes once per nomination). Ties go to the lower class index."""
    preds = np.asarray(probs)[np.asarray(nominated_rows)].argmax(axis=2)        # (n_nominations, N)
    C = np.asarray(probs).shape[2]
    votes = np.stack([(preds == c).sum(axis=0) for c in range(C)], axis=1)     # (N, C)
    return votes.argmax(axis=1)


def agreement_score(pred: np.ndarray, reference: np.ndarray) -> float:
    """Class-balanced agreement: mean over the classes present in the reference of the fraction of the reference's
    samples of that class on which the checkpoint predicts the same class."""
    classes = np.unique(reference)
    return float(np.mean([np.mean(pred[reference == c] == c) for c in classes]))


def select(table: pd.DataFrame, probs: np.ndarray, directions: dict):
    """Full method. Returns (selected row, agreement scores of all checkpoints, reference, nominations)."""
    table = table.reset_index(drop=True)
    probs = np.asarray(probs)
    assert len(table) == probs.shape[0], 'one prediction array per table row'
    nominations = nominate(table, directions)
    reference = agreement_reference(probs, nominations['row'].to_numpy())
    preds = probs.argmax(axis=2)
    scores = np.array([agreement_score(p, reference) for p in preds])
    return int(np.argmax(scores)), scores, reference, nominations
