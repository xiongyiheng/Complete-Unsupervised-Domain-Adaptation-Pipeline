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

**Note:** A `SourceOnly` (`source_only.py`) baseline is also included. 

### Acknowledgements
The algorithm implementations in this codebase are based on their respective original papers and official codebases, as well as the excellent **[UDABench_ECCV2024](https://github.com/ViLab-UCSD/UDABench_ECCV2024)** repository. We thank the authors for their open-source contributions.
