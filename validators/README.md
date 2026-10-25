## Validators

Our benchmark implements a diverse set of validators for UDA model selection, spanning various evaluation paradigms:

| Category | Validators |
| :--- | :--- |
| **Source-Guided** | Source Accuracy (validation set) (`src_acc.py`), IWCV (`iwcv.py`), DEV (`dev.py`) |
| **Target-Based** | Entropy (`entropy.py`), InfoMax (`infomax.py`), Corr-C (`corrc.py`), BNM(V) (`bnm_v.py`), MCC(V) (`mcc_v.py`), SND (`snd.py`), ClassAMI (`classami.py`), MixVal (`mixval.py`), TransferScore (`transferscore.py`) |

**Note:** An `Oracle` (`tgt_oracle.py`) validator (using target labels to select) is also included to serve as an upper-bound target performance.

### Saved Checkpoint Structure

Since researchers and codebases save model checkpoints in many different ways, we explicitly outline our `.pt` file structure below. This clarifies exactly what data inputs each validator script expects and uses to calculate its selection scores.
 
Checkpoints are written with `utils/checkpoint.py::save_checkpoint(path, model, loaders, device, **meta)`. Besides the model outputs for the four data splits, each checkpoint stores the **model weights**, so that validators that must re-run the model (MixVal) or need the classifier itself (TransferScore) are computed as in their original papers rather than approximated from stored outputs.

Each split (`'source_train'`, `'target_train'`, `'source_test'`, `'target_test'`; loaders with `shuffle=False`) contains:
*   **`features`**: The representations extracted by the model's backbone (concatenated over backbones for CoUDA).
*   **`head_features`**: The input of the final linear layer of each classifier head, `(N, D)` or `(H, N, D)` for models with `H` heads (MCD, CoUDA).
*   **`logits`**: The final predictions (`model.predict(x)`; the averaged prediction for two-head models).
*   **`gts`**: The corresponding ground truth labels.

**Structure Overview:**
```python
checkpoint = {
    'state_dict':   model.state_dict(),        # restore with utils/checkpoint.py::load_weights(model, checkpoint)
    'head_weights': [W_1, ...],                # weight of the final linear layer of each classifier head (C x D)
    'meta':         {...},                     # e.g. iteration, lambda, fold
    'source_train': {'features': tensor, 'head_features': tensor, 'logits': tensor, 'gts': tensor},
    'target_train': {...},
    'source_test':  {...},
    'target_test':  {...},
}
```

**Validators that use the weights:**
*   **MixVal** (`mixval.py`): official ICE-inter — target samples in a fixed shuffled order, each paired with its reverse within batches of 128, mixed **in input space** with λ = 0.55 and passed through the restored model; accuracy w.r.t. the dominant pseudo label on inter-class pairs.
*   **TransferScore** (`transferscore.py`): Hopkins statistic on `head_features`, batch-averaged information-maximization loss, and the uniformity of the true final-layer weights `head_weights` (averaged over heads for two-head models), on the target training split.

### Acknowledgements
The validator implementations in this codebase are based on their respective original papers and official codebases. We thank the authors for their open-source contributions and making their methodologies publicly available.
