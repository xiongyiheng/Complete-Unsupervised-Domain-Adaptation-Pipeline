## Validators

Our benchmark implements a diverse set of validators for UDA model selection, spanning various evaluation paradigms:

| Category | Validators |
| :--- | :--- |
| **Source-Guided** | Source Accuracy (validation set) (`src_acc.py`), IWCV (`iwcv.py`), DEV (`dev.py`) |
| **Target Certainty** | Entropy (`entropy.py`) |
| **Target Certainty + Diversity** | InfoMax (`infomax.py`), Corr-C (`corrc.py`), BNM(V) (`bnm_v.py`), MCC(V) (`mcc_v.py`) |
| **Target Neighbor Structure** | SND (`snd.py`) |

**Note:** An `Oracle` (`tgt_oracle.py`) validator (using target labels to select) is also included to serve as an upper-bound target performance.

### Saved Checkpoint Structure

Since researchers and codebases save model checkpoints in many different ways, we explicitly outline our `.pt` file structure below. This clarifies exactly what data inputs each validator script expects and uses to calculate its selection scores.
 
Each checkpoint contains four primary keys corresponding to the data splits: `'source_train'`, `'target_train'`, `'source_test'`, and `'target_test'`. Every split contains a nested dictionary with the following elements:
*   **`features`**: The representations extracted by the model's backbone.
*   **`logits`**: The final predictions output by the classifier head.
*   **`gts`**: The corresponding ground truth labels.
 
**Structure Overview:**
```python
checkpoint = {
    'source_train': {
        'features': tensor,  # Backbone outputs
        'logits':   tensor,  # Classifier outputs
        'gts':      tensor   # Ground truth labels
    },
    'target_train': { 'features': tensor, 'logits': tensor, 'gts': tensor },
    'source_test':  { 'features': tensor, 'logits': tensor, 'gts': tensor },
    'target_test':  { 'features': tensor, 'logits': tensor, 'gts': tensor }
}

### Acknowledgements
The validator implementations in this codebase are based on their respective original papers and official codebases. We thank the authors for their open-source contributions and making their methodologies publicly available.
