# Fundus Dataset Setup

This directory outlines the data source and experimental setup for the Fundus dataset used in our framework. Unlike the other modalities, the source and target domains here come from the **same patient cohort**, imaged with two different modalities:

* **SLO:** Scanning Laser Ophthalmoscopy fundus images.
* **OCT:** En-face (OCT-derived) fundus images.

## 1. Data Acquisition & Experimental Setup

Data comes from the **FairDomain** dataset, released by the Harvard Ophthalmology AI Lab:
[Harvard-Ophthalmology-AI-Lab/FairDomain](https://github.com/Harvard-Ophthalmology-AI-Lab/FairDomain)

**Task Definition:**
We focus on the binary classification between **Glaucoma** and **Non-Glaucoma** cases.

## 2. Reproducibility

We **directly use the official train/test splits** provided by the FairDomain repository.
