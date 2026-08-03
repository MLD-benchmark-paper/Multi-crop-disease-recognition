# MLD: Multi-crop Leaf Disease Benchmark

[![DOI](https://zenodo.org/badge/DOI/XXXXX/zenodo.XXXXXXX.svg)](https://doi.org/XXXXX/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MLD is a unified multi-region benchmark for multi-crop leaf disease recognition. It combines six public crop-disease datasets from the USA, Asia, and Africa into a shared hierarchical taxonomy spanning **18 crops** and **56 crop-disease classes** (including one healthy class per crop), totaling **167,427 images**.

This repository contains the taxonomy mapping, label harmonization code, fixed evaluation splits, and training/evaluation code needed to reproduce the results in our EUVIP 2026 paper, *"Multi-Crop Leaf Disease Recognition: A Unified Benchmark and Cross-Region Study."*

> **Note on images:** We do not redistribute the raw images from the six source datasets. This repository provides the taxonomy, label mapping, and split indices needed to reconstruct MLD from the original sources — see [Source Datasets](#source-datasets) below.

---

## Table of Contents

- [Overview](#overview)
- [Source Datasets](#source-datasets)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Reconstructing MLD](#reconstructing-mld)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

Deep learning models for crop leaf disease recognition routinely report near-perfect in-domain accuracy, but this rarely reflects deployment performance under realistic cross-region domain shift. MLD addresses this by:

1. Unifying six geographically diverse crop-disease datasets into a single crop–disease taxonomy.
2. Defining standardized **single-source** and **pooled multi-source** evaluation protocols that explicitly probe cross-region generalization.
3. Comparing a **flat classifier** against a **hierarchical formulation (HiLeaD)** that conditions disease prediction on the predicted crop.

## Source Datasets

| Dataset | Origin | Samples | Crops | License / Access |
|---|---|---|---|---|
| [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) | USA | 54,306 | 14 | See source repository |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) | India | 2,569 | 13 | See source repository |
| [CCMT](https://data.mendeley.com/) | Ghana | 24,881 | 4 | See source repository |
| [MAK (Makerere)](https://doi.org/10.7910/DVN/LPGHKK) | Uganda | 46,156 | 3 | Harvard Dataverse |
| [NMAIST](https://doi.org/10.7910/DVN/LQUWXW) | Tanzania | 34,345 | 2 | Harvard Dataverse |
| [PSFD-Musa](https://doi.org/10.1016/j.dib.2022.108427) | India | 5,170 | 1 | See source repository |

> ⚠️ Please review and comply with each source dataset's individual license before use. Links above point to the original hosting locations; verify URLs are current before downloading.

## Repository Structure

```
Multi-crop-disease-recognition/
├── data/
│   ├── taxonomy.json          # crop → disease hierarchy (C, D_c structure)
│   ├── label_mapping.json     # source dataset labels → unified taxonomy
│   ├── source_datasets.md     # download links, licenses, acquisition notes
│   └── splits/                # fixed 80/20 stratified split indices (per source)
├── src/
│   ├── data_prep/              # label harmonization, splitting, per-class capping
│   ├── models/                 # flat classifier, HiLeaD
│   ├── train.py
│   └── evaluate.py
├── results/
│   └── per_fold_scores.csv     # per-fold Crop F1 / Disease F1 (5-fold CV)
└── docs/
    └── reproducing_paper_results.md
```

## Setup

```bash
git clone https://github.com/MLD-benchmark-paper/Multi-crop-disease-recognition.git
cd Multi-crop-disease-recognition
conda env create -f environment.yml
conda activate mld-benchmark
```

## Reconstructing MLD

1. Download each source dataset from the links in [Source Datasets](#source-datasets).
2. Run the harmonization script to map each source's native labels onto the shared MLD taxonomy:

   ```bash
   python src/data_prep/harmonize_labels.py \
       --source plantvillage \
       --raw_dir /path/to/plantvillage \
       --mapping data/label_mapping.json \
       --out_dir data/processed/plantvillage
   ```

3. Repeat for each of the six source datasets.
4. Use the provided split files in `data/splits/` to reproduce the exact train/test partitions used in the paper — do not re-split, as this will not match reported results.

## Reproducing Paper Results

```bash
# Train HiLeaD on the pooled MLD training set (Setup A, lr=1e-4)
python src/train.py --config src/config/setup_a.yaml --model hileaD

# Evaluate on all six held-out regional test sets
python src/evaluate.py --model hileaD --checkpoint <path_to_checkpoint>
```

See [`docs/reproducing_paper_results.md`](docs/reproducing_paper_results.md) for the full protocol, including flat-classifier baselines and single-source training runs.

## Results

Per-fold Crop F1 and Disease F1 scores underlying Table III and Table IV of the paper are provided in [`results/per_fold_scores.csv`](results/per_fold_scores.csv) for downstream statistical analysis (e.g., significance testing between flat and HiLeaD).

## Citation

If you use MLD in your research, please cite:

```bibtex
@inproceedings{nalwanga2026mld,
  title     = {Multi-Crop Leaf Disease Recognition: A Unified Benchmark and Cross-Region Study},
  author    = {Nalwanga, Rosemary and [co-authors]},
  booktitle = {Proceedings of the European Workshop on Visual Information Processing (EUVIP)},
  year      = {2026}
}
```

A `CITATION.cff` file is also included for GitHub's native citation support.

Please additionally cite the original source datasets you use — see [Source Datasets](#source-datasets) for their respective papers.

## License

Code, taxonomy mapping, and split indices in this repository are released under the [MIT License](LICENSE). This license applies only to the contents of this repository, not to the source dataset images, which remain subject to their original licenses.
