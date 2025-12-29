# TrustScore — Confidence-Aware ML with Calibration & Trust Modeling

> **A production-style machine learning system that learns *when* to trust predictions not just *what* to predict.**

---

## Overview

Most ML classifiers output probabilities, but those probabilities are often **miscalibrated**. A model claiming "90% confidence" may be wrong far more than 10% of the time. In real-world systems, this leads to over-automation, hidden risk, and poor decision-making.

**TrustScore** addresses this by building a confidence-aware ML pipeline that:

- Trains baseline classifiers
- Calibrates predicted probabilities
- Learns a secondary **trust model** to predict correctness
- Explicitly trades off **coverage vs accuracy**
- Tracks everything with **Metaflow + MLflow**

This enables systems that can decide:
> *Should this prediction be automated, or should it be deferred?*

---

## Key Concepts

### 1. Base Prediction Models
Standard supervised classifiers (e.g., Logistic Regression, Random Forest) trained and selected using validation PR-AUC.

### 2. Probability Calibration
Predicted probabilities are calibrated to better reflect true likelihoods using:
- Sigmoid calibration
- Isotonic regression

Calibration quality is measured with:
- Expected Calibration Error (ECE)
- Brier Score

### 3. Trust Modeling (Meta-Model)
A secondary model is trained to estimate whether a prediction is likely to be **correct**.

Key properties:
- Trained using **out-of-fold (OOF)** calibrated probabilities
- Avoids train-test leakage
- Mimics real deployment behavior

### 4. Coverage–Accuracy Tradeoff
Instead of a single accuracy number, the system evaluates:
- Accuracy at different trust thresholds
- Fraction of predictions that can be automated at each threshold

This is critical for **human-in-the-loop** and high-stakes ML systems.

---

## System Architecture

```
Raw Data
  |
  v
Preprocessing
  |
  v
Base Models (foreach)
  |
  v
Join -> Best Base Model
  |
  v
Calibration (foreach)
  |
  v
Join -> Best Calibration
  |
  v
OOF Calibrated Predictions (TRAIN)
  |
  v
Trust Model Training
  |
  v
Final Test Evaluation (Coverage vs Accuracy)
```

---

## Tech Stack

- Python 3.13
- scikit-learn
- Metaflow (workflow orchestration)
- MLflow (experiment tracking)
- NumPy / Pandas

---

## Metrics Tracked

### Base Model
- PR-AUC
- ROC-AUC

### Calibration
- Expected Calibration Error (ECE)
- Brier Score

### Trust Model
- Trust AUC (correct vs incorrect prediction detection)

### System-Level
- Coverage vs Accuracy curves
- Automation rate at different trust thresholds

All metrics, parameters, and artifacts are logged to **MLflow**.

---

## Project Structure

```
.
├── flow.py          # Metaflow pipeline
├── src/
│   ├── data/        # Data loading & splitting
│   ├── features/    # Preprocessing & trust features
│   ├── models/      # Base models, calibration, OOF logic
│   └── evaluation/  # Calibration & trust metrics
├── artifacts/       # Generated plots & tables
└── readme.md
```

---

## Run the full pipeline

```bash
python flow.py run
```
