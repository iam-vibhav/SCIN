# SCIN - Skin Condition Identification Network

A multimodal machine learning pipeline for skin disease classification using both clinical metadata and dermatology images from the SCIN dataset.

The project combines:
- LightGBM for tabular clinical metadata
- A custom Siamese CNN for multi-image feature extraction
- Late fusion using a feedforward neural network

The system is designed for:
- Multi-label skin condition prediction
- Soft-label probability learning
- Multi-image patient cases
- Missing image handling

---

# Features

- End-to-end multimodal pipeline
- Metadata + image fusion
- Multi-label soft classification
- Custom Siamese CNN architecture
- Missing image masking support
- Multilabel stratified train/test split
- Late fusion meta learner
- Soft-label learning using KL divergence
- GPU-compatible TensorFlow pipeline

---

# Dataset

Dataset used:
- SCIN Dataset (Google Health)

Sources:
- Clinical metadata
- Dermatologist annotations
- Up to 3 dermatology images per case

Dataset characteristics:
- Originally 370 skin condition classes
- Rare labels (<50 samples) removed during preprocessing
- Multi-label weighted diagnosis distributions preserved

---

# Project Architecture

## 1. Metadata Pipeline

Clinical metadata is cleaned, encoded, and passed into several machine learning models.

### Models Tested
- Ridge Regressor
- LightGBM Regressor
- KNN Regressor
- Elastic Net
- Feedforward Neural Network

### Best Metadata Model
LightGBM performed best on structured clinical features.

---

## 2. Image Pipeline

The image branch uses a custom Siamese CNN that processes up to 3 images per patient case.

### CNN Features
- Shared CNN encoder
- Multi-image aggregation
- Missing image masking
- Weighted feature averaging
- Softmax probability outputs

### Encoder Architecture

- Conv2D(32)
- Conv2D(64)
- Conv2D(128)
- GlobalAveragePooling2D

### Additional Experiments
- EfficientNetB0 feature extractor
- Fine-tuned EfficientNetB0

### Best Image Model
The custom Siamese CNN achieved the best overall performance for this dataset.

---

## 3. Fusion Pipeline

The final model combines:
- Metadata features
- CNN image embeddings

using late fusion.

### Fusion Strategy
1. Extract image embeddings from the Siamese CNN
2. Concatenate image and metadata features
3. Pass combined features into a dense meta learner
4. Predict final soft-label probabilities

---

# Performance

## Metadata Models

| Model | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Ridge Regressor | 0.3583 | 0.6358 | 0.7697 |
| LightGBM | 0.3621 | 0.6481 | 0.7695 |
| KNN Regressor | 0.3366 | 0.6063 | 0.6811 |
| Elastic Net | 0.3524 | 0.6476 | 0.7697 |
| Feedforward Neural Network | 0.3346 | 0.6260 | 0.7520 |

---

## Image Models

| Model | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Custom Siamese CNN | 0.3745 | 0.6276 | 0.7243 |
| EfficientNetB0 Feature Extractor | 0.3724 | 0.6914 | 0.8045 |
| EfficientNetB0 Transfer Learning | 0.3307 | 0.6319 | 0.7677 |

---

## Final Fusion Model

| Metric | Score |
|---|---|
| Top-1 Accuracy | 0.3745 |
| Top-3 Accuracy | 0.6543 |
| Top-5 Accuracy | 0.7963 |
| Mean KL Divergence | 2.2603 |
| Argmax Accuracy | 0.2202 |



---

# Installation

## Clone Repository

```
git clone <repository-url>
cd SCIN
```


---

# Main Dependencies

```
tensorflow
scikit-learn
lightgbm
numpy
pandas
matplotlib
iterative-stratification
tqdm
```

Tested on:
- Python 3.10+
- TensorFlow 2.x

---



The notebook contains:
- Data preprocessing
- Feature engineering
- Metadata modelling
- CNN training
- Fusion training
- Evaluation
- Artifact exporting

---

# Data Preprocessing

The preprocessing pipeline includes:
- Rare class filtering
- Missing image handling
- Feature encoding
- Leakage removal
- Soft-label generation
- Multilabel stratified splitting

---

# Image Handling

Each case may contain:
- 1 image
- 2 images
- 3 images

Missing images are automatically:
- Replaced with zero tensors
- Ignored during feature aggregation using masking

---

# Training Details

## Loss Function
KL Divergence Loss

Used because labels are soft probability distributions instead of hard one-hot labels.

---

# Evaluation Metrics

- Top-1 Accuracy
- Top-3 Accuracy
- Top-5 Accuracy
- Mean Squared Error
- Mean Absolute Error
- KL Divergence

---

# Exported Artifacts

The project exports:
- `lgbm_model.pkl`
- `meta_model.keras`
- `baseline_model.keras`
- `scaler.pkl`
- `feature_columns.json`
- `label_columns.json`

All artifacts can be downloaded by running the notebook (the artifacts were too large to upload to github 😅):


---

# Technical Highlights

## Soft Label Learning

Instead of converting diagnoses into hard labels, the project preserves dermatologist confidence distributions.

Example:

```
{
    "eczema": 0.7,
    "psoriasis": 0.3
}
```

---

## Missing Image Aware CNN

The Siamese CNN dynamically masks missing images before feature aggregation.

This allows robust handling of incomplete image sets.

---

## Late Fusion

Instead of early feature fusion, the project uses:
- Independent modality learning
- Feature-level concatenation
- Meta-learner based final prediction

This improved overall performance compared to standalone models.
