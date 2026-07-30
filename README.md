# Complete Unsupervised Domain Adaptation (UDA) Pipeline

A complete pipeline for **unsupervised domain adaptation (UDA)** on medical imaging tasks, covering both (1) **adaptation** — training UDA algorithms jointly on a labeled source domain (e.g. one hospital) and an unlabeled target domain (e.g. a different one), and (2) **label-free model selection** — since the target domain has no labels, validators are used to select the best checkpoint using only source labels and unlabeled target data.

## Repository Structure

| Folder | Contents |
| :--- | :--- |
| [`algorithms/`](algorithms/README.md) | UDA training algorithms (DANN, CDAN, MMD, MCC, BNM, ATDOC, MCD, DALN, AD2A, CoUDA) plus a `SourceOnly` baseline. |
| [`backbones/`](backbones/README.md) | Feature extractor backbones — 2D CNNs/ViTs/MLPs (via `timm`) for X-ray, and 3D ResNet for volumetric MRI/CT. |
| [`validators/`](validators/README.md) | Model-selection metrics for picking the best checkpoint without target labels (source accuracy, IWCV, DEV, entropy, InfoMax, SND, etc.), plus an `Oracle` upper-bound validator. |
| [`datasets/`](datasets) | Data acquisition, preprocessing, and cross-validation split instructions for each imaging modality: [`brain_mri/`](datasets/brain_mri/README.md) (ADNI, AIBL), [`cxr/`](datasets/cxr/README.md) (RSNA, Child-Xray, LDD, CRD), and [`fundus/`](datasets/fundus/README.md) (FairDomain: SLO, OCT). |
| `utils/` | Shared helper functions used across the pipeline. |

## How It Fits Together

1. **Train** — pick a backbone from `backbones/` and an algorithm from `algorithms/` to adapt a model from a labeled source dataset to an unlabeled target dataset.
2. **Checkpoint** — save model outputs (features, logits, ground truth) for the source/target train/test splits, following the `.pt` structure documented in [`validators/README.md`](validators/README.md).
3. **Validate** — since target labels aren't available for real model selection, use a validator from `validators/` to score and pick the best checkpoint across training runs/hyperparameters.

## Task Definitions

* **Brain MRI:** binary classification of Alzheimer's Disease (AD) vs. Cognitively Normal (CN).
* **Chest X-Ray:** binary classification of Pneumonia vs. Non-Pneumonia.
* **Fundus:** binary classification of Glaucoma vs. Non-Glaucoma.

See each dataset's README for data sources, preprocessing steps, and reproducibility splits.

## Citation

If you find this code helpful, please consider giving our code repository a star ⭐️
