"""
Two-level agreement selection (Xiong et al., MICCAI Workshop 2026).

Inputs (one row per candidate checkpoint, same target samples for every checkpoint):
    table   pandas DataFrame with a column 'algorithm' and one column per validator score
    probs   array (K, N, C): target predicted probabilities of the K checkpoints, in table row order
    directions  {validator: 'max' | 'min'}  (whether a higher or a lower score is better)

Level 1  For every algorithm a and validator v, v nominates its best checkpoint(s) of a (all checkpoints tied at
         the best score form one nomination).
Level 2  The agreement reference y~ is the class with the highest mean predicted probability over the nominations
         (each nomination weighs one, its tied checkpoints are averaged).
Select   Every checkpoint of the pool is scored by its class-balanced agreement with y~; the best one is deployed
         (exact ties: all tied checkpoints are returned).
"""
import numpy as np
import pandas as pd


def nominate(table: pd.DataFrame, directions: dict) -> list:
    """Level 1: one nomination per (algorithm, validator), as an array of the table rows tied at the best score.
    Checkpoints without a score (NaN) are skipped."""
    nominations = []
    for algo, sub in table.groupby('algorithm', sort=True):
        for v, d in directions.items():
            s = sub[v].to_numpy(dtype=float)
            ok = np.isfinite(s)
            if not ok.any():
                continue
            s = np.where(ok, s if d == 'max' else -s, -np.inf)
            nominations.append(sub.index.to_numpy()[s == s.max()])
    return nominations


def agreement_reference(probs: np.ndarray, nominations: list) -> np.ndarray:
    """Level 2: argmax of the mean over nominations of each nomination's mean predicted probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    return np.mean([probs[np.asarray(n)].mean(axis=0) for n in nominations], axis=0).argmax(axis=1)


def agreement_score(pred: np.ndarray, reference: np.ndarray) -> float:
    """Class-balanced agreement: mean over the classes present in the reference of the fraction of the reference's
    samples of that class on which the checkpoint predicts the same class."""
    classes = np.unique(reference)
    return float(np.mean([np.mean(pred[reference == c] == c) for c in classes]))


def select(table: pd.DataFrame, probs: np.ndarray, directions: dict):
    """Full method. Returns (selected rows -- several only for exact ties --, agreement scores of all checkpoints,
    reference, nominations)."""
    table = table.reset_index(drop=True)
    probs = np.asarray(probs)
    assert len(table) == probs.shape[0], 'one prediction array per table row'
    nominations = nominate(table, directions)
    reference = agreement_reference(probs, nominations)
    preds = probs.argmax(axis=2)
    scores = np.array([agreement_score(p, reference) for p in preds])
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=1e-9, atol=1e-12))
    return best, scores, reference, nominations
