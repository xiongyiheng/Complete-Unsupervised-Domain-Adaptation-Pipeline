## UDA Algorithms

Our benchmark implements a diverse set of UDA algorithms, spanning various adaptation paradigms:

| Category | Algorithms |
| :--- | :--- |
| **Feature Distribution** | MMD (`mmd.py`) |
| **Adversarial Alignment** | DANN (`dann.py`), CDAN (`cdan.py`), DALN (`daln.py`) |
| **Information Maximization** | MCC (`mcc.py`) |
| **SVD Loss** | BNM (`bnm.py`) |
| **Pseudo Labeling** | ATDOC (`atdoc_nc.py`, `atdoc_na.py`)* |
| **Classifier Discrepancy** | MCD (`mcd.py`) |
| **Modality-Specific** | AD2A (`ad2a.py`) (Brain MRI), CoUDA (`couda.py`) (Chest X-Ray) |

*\* For ATDOC, **NC** stands for Nearest Centroid Classifier and **NA** stands for Neighbor Aggregation.*

**Training loop.** Every algorithm returns `(total_loss, cls_loss, transfer_loss)` from `compute_loss(...)` and provides `predict(x)` for evaluation and checkpointing. Exceptions:
* **MCD** performs its three-step update internally with its own optimizers: call `update(x_s, y_s, x_t, ce_weight)`.
* **ATDOC** follows the official training loop ([tim-learn/ATDOC](https://github.com/tim-learn/ATDOC)): label-smoothed source loss, target loss ramped by `iter / max_iters`, and a memory refreshed **after** the optimizer step:
  ```python
  loss, _, _ = model.compute_loss(x_s, y_s, x_t, ce_weight, tgt_index)   # ATDOC_NC: no tgt_index
  optimizer.zero_grad(); loss.backward(); optimizer.step()
  model.update_memory(x_t, tgt_index)                                       # ATDOC_NC: update_memory(x_t)
  ```
  `tgt_index` is the position of each target sample in the target training set (the target loader must return it).
* **AD2A** uses only the classification loss for the first `warmup_steps` iterations; its `transfer_loss_weight` (λ ∈ {0.1, 0.5, 1.0}) selects the preset adversarial / attention-alignment weights.

**Note:** A `SourceOnly` (`source_only.py`) baseline is also included. 

### Acknowledgements
The algorithm implementations in this codebase are based on their respective original papers and official codebases, as well as the excellent **[UDABench_ECCV2024](https://github.com/ViLab-UCSD/UDABench_ECCV2024)** repository. We thank the authors for their open-source contributions.
