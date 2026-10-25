"""
Build the two inputs of the method from saved checkpoints:

    scores.csv          one row per checkpoint: algorithm, lambda, iteration, path, one column per validator
    target_probs.npy    (K, N, C) target predictions (softmax), rows in the order of scores.csv

Checkpoints are expected in the format of the Complete UDA Pipeline (utils/checkpoint.py::save_checkpoint), one file
per checkpoint, e.g.  <root>/<ALGORITHM>_lam<LAMBDA>/ckpt_<ITERATION>.pt.  Validators are taken from this repository's
validators/ folder.  The validators shown here only need the stored outputs; MixVal and
TransferScore additionally need the restored model and target inputs (see the pipeline README) and can be added as
further columns in the same way.

Usage:
    python build_score_table.py --ckpt-root runs/
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# validator name -> (direction, function of the checkpoint dict)
def validator_table():
    from entropy import entropy
    from infomax import infomax
    from bnm_v import bnm
    from mcc_v import mcc
    from corrc import corrc_prob
    from snd import snd
    from classami import classami
    from dev import compute_dev
    from iwcv import iwcv
    from utils import compute_accuracy

    np_ = lambda t: t.numpy() if torch.is_tensor(t) else np.asarray(t)
    tt = lambda ck, k: np_(ck['target_test'][k])
    src = lambda ck: (np_(ck['source_train']['features']), np_(ck['target_train']['features']),
                      np_(ck['source_test']['features']), np_(ck['source_test']['logits']),
                      np_(ck['source_test']['gts']).reshape(-1))
    return {
        'Source-Risk': ('max', lambda ck: compute_accuracy(np_(ck['source_test']['logits']), np_(ck['source_test']['gts']))),
        'IWCV':        ('min', lambda ck: iwcv(*src(ck))),
        'DEV':         ('min', lambda ck: compute_dev(*src(ck))[0]),
        'DEV-N':       ('min', lambda ck: compute_dev(*src(ck))[1]),
        'Entropy':     ('min', lambda ck: entropy(tt(ck, 'logits'))),
        'InfoMax':     ('max', lambda ck: infomax(tt(ck, 'logits'))),
        'Corr-C':      ('max', lambda ck: corrc_prob(tt(ck, 'logits'))),
        'MCC (V)':     ('min', lambda ck: mcc(tt(ck, 'logits'))),
        'BNM (V)':     ('max', lambda ck: bnm(tt(ck, 'logits'))),
        'SND':         ('min', lambda ck: snd(tt(ck, 'logits'))),
        'Class-AMI':   ('max', lambda ck: classami(tt(ck, 'logits'), tt(ck, 'features'))),
    }


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--pipeline', default=str(Path(__file__).resolve().parent.parent),
                    help='repository root (contains validators/)')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.pipeline) / 'validators'))
    vals = validator_table()

    rows, probs = [], []
    for path in sorted(Path(a.ckpt_root).glob('*_lam*/ckpt_*.pt')):
        m = re.match(r'(.+)_lam([0-9.]+)$', path.parent.name)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        row = dict(algorithm=m.group(1), lam=float(m.group(2)), iteration=int(path.stem.split('_')[1]), path=str(path))
        for name, (_, fn) in vals.items():
            row[name] = fn(ck)
        rows.append(row)
        probs.append(softmax(np.asarray(ck['target_test']['logits'], dtype=np.float64)).astype(np.float32))

    out = Path(a.out)
    pd.DataFrame(rows).to_csv(out / 'scores.csv', index=False)
    np.save(out / 'target_probs.npy', np.stack(probs))
    pd.Series({k: d for k, (d, _) in vals.items()}, name='direction').to_csv(out / 'directions.csv')
    print(f'{len(rows)} checkpoints -> {out / "scores.csv"}, {out / "target_probs.npy"}, {out / "directions.csv"}')


if __name__ == '__main__':
    main()
