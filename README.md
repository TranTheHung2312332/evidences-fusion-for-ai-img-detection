# Image Authenticity Research

An early-stage research project exploring AI-generated image detection through three complementary forms of evidence: **semantic consistency**, **image forensics**, and **provenance information**.

The repository currently contains scripts for preparing evaluation datasets and generating provenance data. Model implementations and experimental results will be added as the project develops.

## Repository structure

```text
scripts/
├── build_dataset(2).py
└── provenance_augmentation(1).py
```

- **`build_dataset(2).py`** — Builds reproducible, balanced validation and test sets from real and AI-generated image sources. It also produces source manifests and standardized metadata.
- **`provenance_augmentation(1).py`** — Audits available image provenance (including EXIF, XMP, and C2PA) and prepares controlled provenance scenarios and metadata views for evaluation.

## Status

**Work in progress.** The current repository covers dataset construction and provenance preparation only. Documentation, model pipelines, and evaluation results will be added later.

The source datasets remain subject to their respective licenses and terms of use.
