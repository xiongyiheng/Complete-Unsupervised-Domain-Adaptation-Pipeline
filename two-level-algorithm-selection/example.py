"""
Run the two-level agreement selection.

    python example.py                                   # synthetic pool (no data needed)
    python example.py --scores scores.csv --probs target_probs.npy --directions directions.csv
                                                        # files written by build_score_table.py
"""
import argparse

import numpy as np
import pandas as pd

from two_level_selection import select


def synthetic_pool(seed=0, n_target=300):
    """3 algorithms x 3 lambdas x 10 iterations of a binary task. Each checkpoint has a hidden quality q; its
    predictions are correct with probability q, and every validator sees q only through its own noise."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n_target)                            # target labels: used ONLY to report accuracy below
    directions = {'Source-Risk': 'max', 'Entropy': 'min', 'InfoMax': 'max', 'DEV': 'min'}
    rows, probs = [], []
    for algo, base in [('DANN', 0.78), ('MCC', 0.84), ('BNM', 0.81)]:
        for lam in (0.1, 0.5, 1.0):
            for it in range(1000, 11000, 1000):
                q = np.clip(base + 0.05 * rng.randn() + 0.01 * it / 1000, 0.55, 0.97)
                correct = rng.rand(n_target) < q
                pred = np.where(correct, y, 1 - y)
                p1 = np.clip(np.where(pred == 1, 0.6 + 0.4 * rng.rand(n_target), 0.4 * rng.rand(n_target)), 1e-3, 1 - 1e-3)
                probs.append(np.stack([1 - p1, p1], axis=1))
                row = dict(algorithm=algo, lam=lam, iteration=it)
                for v, d in directions.items():
                    noisy = q + 0.08 * rng.randn()
                    row[v] = noisy if d == 'max' else -noisy
                rows.append(row)
    return pd.DataFrame(rows), np.stack(probs), directions, y


def balanced_acc(pred, y):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores'); ap.add_argument('--probs'); ap.add_argument('--directions')
    a = ap.parse_args()
    if a.scores:
        table, probs = pd.read_csv(a.scores), np.load(a.probs)
        directions = pd.read_csv(a.directions, index_col=0)['direction'].to_dict()
        y = None
    else:
        table, probs, directions, y = synthetic_pool()

    best, scores, reference, nominations = select(table, probs, directions)

    print(f'{len(table)} checkpoints, {table.algorithm.nunique()} algorithms, {len(directions)} validators, '
          f'{len(nominations)} nominations')
    for b in best:
        print('selected checkpoint:', table.loc[b].drop([c for c in directions if c in table.columns]).to_dict(),
              f'agreement score {scores[b]:.3f}')

    if y is not None:  # synthetic demo: compare with the labels the method never saw
        acc = np.array([balanced_acc(p.argmax(1), y) for p in probs])
        print(f'target accuracy  selected {acc[best].mean():.3f} | best checkpoint (oracle) {acc.max():.3f}')
        for v, d in directions.items():
            s = table[v].to_numpy()
            print(f'  single validator {v:12s} picks {acc[np.argmax(s) if d == "max" else np.argmin(s)]:.3f}')


if __name__ == '__main__':
    main()
