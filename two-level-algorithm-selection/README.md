# Towards Practical Algorithm Selection for Unsupervised Domain Adaptation in Medical Imaging
MICCAI 2026 Workshop

Yiheng Xiong, [Luisa Gallée](https://scholar.google.de/citations?user=RotwTNsAAAAJ&hl=en), [Daniel Santak Wolf](https://viscom.uni-ulm.de/members/daniel-wolf/), [Heiko Hillenhagen](https://www.linkedin.com/in/heiko-hillenhagen/) and [Michael Götz](https://xrad-ulm.github.io/)

A **label-free criterion for jointly selecting the UDA algorithm and its hyperparameters**. Given a pool of candidate checkpoints from several UDA algorithms, each trained with several hyperparameter configurations, every candidate is scored against an **agreement reference** built from the candidates themselves, and the candidate with the highest score is deployed. No target labels are used.

## Method

1. **Level 1 — within-algorithm nomination.** For every algorithm $a$ and validator $v$, the validator nominates its best-scoring checkpoint of that algorithm, $\theta_{a,v}$.
2. **Level 2 — across-algorithm agreement.** The nominated checkpoints of *all* algorithms vote on every target sample (one vote per nomination); the majority class is the agreement reference $\tilde{y}^t$.
3. **Selection.** Every checkpoint of the pool is scored by its class-balanced agreement with $\tilde{y}^t$ (per-class fraction of matching predictions, averaged over classes). The single best checkpoint is deployed; the nominated checkpoints are only used to build the reference.

## Repository Structure

| File | Contents |
| :--- | :--- |
| [`two_level_selection.py`](two_level_selection.py) | The method: `nominate` (Level 1), `agreement_reference` (Level 2), `agreement_score`, and `select` (all three steps). NumPy/pandas only. |
| [`build_score_table.py`](build_score_table.py) | Builds the method's inputs from saved checkpoints: validator scores per checkpoint (validators from [`../validators/`](../validators/README.md)) and the target predictions. |
| [`example.py`](example.py) | Runs the selection on a synthetic pool, or on the files written by `build_score_table.py`. |

## How It Fits Together

1. **Train and checkpoint** — train several UDA algorithms under several hyperparameter configurations and save checkpoints during training with this repository (algorithms in [`../algorithms/`](../algorithms/README.md), checkpoints via [`../utils/checkpoint.py`](../utils/checkpoint.py)`::save_checkpoint`), one file per checkpoint, e.g. `runs/<ALGORITHM>_lam<LAMBDA>/ckpt_<ITERATION>.pt`.
2. **Score** — compute the validators for every checkpoint and store them, together with the target predictions:
   ```bash
   python build_score_table.py --ckpt-root runs/
   ```
   This writes three files:
   * `scores.csv` — one row per checkpoint: `algorithm, lam, iteration, path`, and one column per validator;
   * `target_probs.npy` — array `(K, N, C)`: softmax predictions of the `K` checkpoints on the `N` target samples, rows in the order of `scores.csv`;
   * `directions.csv` — for every validator whether a higher (`max`) or lower (`min`) score is better.
3. **Select** —
   ```bash
   python example.py --scores scores.csv --probs target_probs.npy --directions directions.csv
   ```
   or in Python:
   ```python
   import numpy as np, pandas as pd
   from two_level_selection import select

   table = pd.read_csv("scores.csv")
   probs = np.load("target_probs.npy")
   directions = pd.read_csv("directions.csv", index_col=0)["direction"].to_dict()

   best, scores, reference, nominations = select(table, probs, directions)   # best: row of the selected checkpoint
   print(table.loc[best, ["algorithm", "lam", "iteration", "path"]])
   ```

Any validator can be used: add a column with its scores to `scores.csv` and its direction to `directions.csv` (e.g. MixVal and TransferScore, which need the restored model and are computed as described in [`../validators/README.md`](../validators/README.md)). Checkpoints with a missing score are skipped by that validator.

**Quick start without data:** `python example.py` builds a synthetic pool (3 algorithms × 3 λ × 10 iterations) and compares the selected checkpoint with the best checkpoint and with single validators.

## Citation

If you find this code helpful, please consider citing:
```BibTeX
@inproceedings{xiong2026towards,
  title={Towards Practical Algorithm Selection for Unsupervised Domain Adaptation in Medical Imaging},
  author={Xiong, Yiheng and Gall{\'e}e, Luisa and Wolf, Daniel Santak and Hillenhagen, Heiko and G{\"o}tz, Michael},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI) Workshops},
  year={2026}
}
```

## Acknowledgement

This study was funded by the German Research Foundation DFG (Project: KEMAI, GRK 3012 – 520750254) and by the German Federal Ministry of Research, Technology and Space BMFTR as part of the University Medicine Network 3.0 (Project: RACOON, 01KX2524).
