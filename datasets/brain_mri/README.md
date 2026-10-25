# Brain MRI Dataset Setup

This directory outlines the data sources,  experimental setup and preprocessing requirements for the Brain MRI datasets used in our framework. 

## 1. Data Acquisition & Experimental Setup

Due to data privacy and usage agreements, raw MRI images must be acquired directly from their respective databases. The data used in this benchmark comes from the following sources:

* **ADNI:** Data can be obtained from [https://adni.loni.usc.edu/data-samples/adni-data/](https://adni.loni.usc.edu/data-samples/adni-data/).
* **AIBL:** Data can be obtained from [https://aibl.csiro.au/adni/index.html](https://aibl.csiro.au/adni/index.html).

**Task Definition:**
We focus on the binary classification bwteen **Alzheimer's Disease (AD)** and **Cognitively Normal (CN)** cases.

## 2. Preprocessing Pipeline

To maintain consistency across the benchmark, all raw scans should be preprocessed. For the preprocess codes, please refer to: [https://github.com/quqixun/BrainPrep](https://github.com/quqixun/BrainPrep).

Please utilize their provided code to apply the following three preprocessing steps:
1. **Registration**
2. **Skull-Stripping**
3. **Bias Field Correction**

## 3. Reproducibility

To facilitate a robust evaluation, we use a stratified five-fold cross-validation scheme on each dataset.
