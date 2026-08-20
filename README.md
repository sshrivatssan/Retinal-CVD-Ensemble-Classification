# Cardiovascular Disease Classification from Retinal Fundus Images using Ensemble Texture Features

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cmpbup.2026.100262-blue)](https://doi.org/10.1016/j.cmpbup.2026.100262)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides the implementation associated with the published work on cardiovascular disease (CVD) classification from retinal fundus images using ensemble texture features and deep neural networks.

The framework integrates **Local Binary Pattern (LBP)** and **Gray-Level Co-occurrence Matrix (GLCM)** texture descriptors with **DenseNet-121, EfficientNet-B3, and MobileNetV3-Large** for classification based on carotid intima-media thickness (CIMT).

## Overview

Cardiovascular diseases (CVDs) are among the leading causes of early mortality worldwide, highlighting the importance of early risk assessment. Retinal fundus imaging provides a non-invasive means of visualizing retinal vascular structures and associated cardiovascular abnormalities.

This work investigates the integration of handcrafted retinal texture descriptors with deep neural networks for CVD classification. LBP captures local retinal texture patterns, while GLCM represents broader spatial texture characteristics. Their combination enables complementary fine- and coarse-scale retinal texture information to be incorporated into deep-learning-based classification.

## Proposed Methodology

### 1. Dataset

The experiments use the **China-Fundus-Carotid Intima-Media Thickness (China-Fundus-CIMT) dataset**, which contains **5,806 bilateral high-resolution retinal fundus images from 2,903 patients**.

Subjects are categorized into two groups according to carotid intima-media thickness:

- **Normal:** CIMT < 0.9 mm
- **Thickened:** CIMT ≥ 0.9 mm

To prevent patient-level information leakage, a **patient-wise stratified 70:15:15 split** is used. Images belonging to the same patient are restricted to a single subset.

| Split | Patients |
|---|---:|
| Training | 1,936 |
| Validation | 483 |
| Testing | 484 |

#### Dataset Access

[China-Fundus-CIMT Dataset](https://springernature.figshare.com/articles/dataset/High-resolution_fundus_images_for_ophthalmomics_and_early_cardiovascular_disease_prediction_China_Fundus_Carotid_Intima-Media_Thickness_dataset/27907056)

The dataset is not redistributed in this repository and should be obtained from its original source.

#### Dataset Citation

For use of the China-Fundus-CIMT dataset, please cite the original dataset publication:

> N. Guo, W. Fu, H. Li, H. Zhang, T. Li, W. Zhang, X. Zhong, T. Pan, F. Sun, A. Gong, et al.,  
> "High-resolution fundus images for ophthalmomics and early cardiovascular disease prediction,"  
> *Scientific Data*, 2025.

### 2. Image Preprocessing

The preprocessing pipeline operates on the green channel of the retinal fundus images to enhance retinal structures and improve contrast.

The main preprocessing stages include:

- Green-channel extraction
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Min-max normalization

The processed retinal images are subsequently used for handcrafted texture feature extraction and deep-learning-based classification.

### 3. Texture Feature Extraction

Two complementary handcrafted texture descriptors are extracted from the retinal fundus images.

#### Local Binary Pattern (LBP)

LBP is used to characterize fine local retinal texture variations by comparing neighbouring pixel intensities with the centre pixel.

Uniform LBP is extracted using:

- Number of sampling points: **P = 16**
- Radius: **R = 2**
- Uniform LBP representation

The resulting representation produces an **18-dimensional LBP feature vector** for each image.

#### Gray-Level Co-occurrence Matrix (GLCM)

GLCM is used to characterize second-order spatial relationships between retinal image intensities. The GLCM features are extracted using:

- **Gray levels:** 64
- **Pixel distances:** 1 and 2
- **Orientations:** 0°, 45°, 90°, and 135°
- **Texture descriptors:** 14 Haralick features

The mean and standard deviation of the 14 Haralick features across the GLCM configurations are used, resulting in a **28-dimensional GLCM feature vector** for each image.

### 4. Ensemble Texture Features

LBP and GLCM capture complementary characteristics of retinal texture, with **LBP capturing fine local texture patterns** and **GLCM capturing broader spatial texture relationships**. The two descriptors are combined to form an **ensemble texture representation**, which is integrated with deep image features for CVD classification.

### 5. Deep Learning Classification

Three deep learning architectures—**DenseNet-121, EfficientNet-B3, and MobileNetV3-Large**—are evaluated using two experimental configurations: **Images** and **Images + LBP + GLCM**. This comparison evaluates the effect of integrating handcrafted retinal texture features with deep neural network representations.

### Proposed Framework

<p align="center">
  <img src="images/block_diagram.png" width="950" alt="Proposed retinal CVD classification framework">
</p>

<p align="center">
  <em>Proposed framework for CVD classification from colour fundus images using ensemble texture features and deep neural networks.</em>
</p>

## Implementation Configuration

The repository implementation uses the following principal training configuration:

| Parameter | Value |
|---|---|
| Input size | 224 × 224 |
| Optimizer | AdamW |
| Initial learning rate | 1 × 10⁻⁴ |
| Batch size | 32 |
| Loss function | Weighted cross-entropy |
| Weight decay | 1 × 10⁻⁴ |
| Maximum epochs | 40 |
| Early-stopping patience | 8 |
| Learning-rate scheduler | ReduceLROnPlateau |

Model selection is based on validation performance before final evaluation on the independent test set.

## Evaluation Metrics

The classification models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)
- Area Under the Receiver Operating Characteristic Curve (AUROC)

## Results

The classification performance reported in the published work is summarized below.

| Input Type | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | MCC | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| Images | DenseNet-121 | 73.30 | 69.60 | 69.30 | 69.40 | 0.389 | 0.767 |
| Images | EfficientNet-B3 | 72.90 | 69.00 | 67.20 | 67.80 | 0.361 | 0.734 |
| Images | MobileNetV3-Large | 69.40 | 65.40 | 65.60 | 65.50 | 0.309 | 0.707 |
| **Images + LBP + GLCM** | **DenseNet-121** | **75.20** | **71.70** | **70.50** | **71.00** | **0.422** | **0.780** |
| Images + LBP + GLCM | EfficientNet-B3 | 72.10 | 68.00 | 66.90 | 67.30 | 0.349 | 0.749 |
| Images + LBP + GLCM | MobileNetV3-Large | 72.90 | 69.10 | 68.50 | 68.80 | 0.376 | 0.766 |

The **DenseNet-121 + LBP + GLCM** configuration achieved the best overall performance, with **75.20% accuracy, 71.00% F1-score, 0.422 MCC, and 0.780 AUROC**. The integration of ensemble texture features improved DenseNet-121 accuracy from **73.30% to 75.20%** and MobileNetV3-Large accuracy from **69.40% to 72.90%**.

## Repository Structure

```text
Retinal-CVD-Ensemble-Classification/
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── training.py
│   └── utils.py
│
├── scripts/
│   ├── create_splits.py
│   ├── evaluate_models.py
│   ├── extract_features.py
│   ├── preprocess.py
│   ├── train_baselines.py
│   └── train_fusion.py
│
├── notebook/
│   └── cvd_identification.ipynb
│
├── images/
│   └── block_diagram.png
│
├── README.md
├── LICENSE
└── .gitignore
```

## Environment

The experiments were implemented in Python using PyTorch and executed with GPU acceleration on an **NVIDIA GeForce RTX 3060**.

Key dependencies include:

- PyTorch
- timm
- OpenCV
- NumPy
- pandas
- scikit-image
- scikit-learn
- Matplotlib

## Publication

This repository accompanies the following publication:

**S. Shrivatssan and Malaya Kumar Nath**,  
"Identification of cardiovascular diseases from colour fundus images using ensemble texture features by deep neural networks,"  
*Computer Methods and Programs in Biomedicine Update*, Volume 10, Article 100262, 2026.

**DOI:** [10.1016/j.cmpbup.2026.100262](https://doi.org/10.1016/j.cmpbup.2026.100262)

## Citation

If you use this repository or the associated methodology in your research, please cite:

```bibtex
@article{shrivatssan2026identification,
  title   = {Identification of cardiovascular diseases from colour fundus images using ensemble texture features by deep neural networks},
  author  = {Shrivatssan, S. and Nath, Malaya Kumar},
  journal = {Computer Methods and Programs in Biomedicine Update},
  volume  = {10},
  pages   = {100262},
  year    = {2026},
  issn    = {2666-9900},
  doi     = {10.1016/j.cmpbup.2026.100262}
}
```

## License

The source code in this repository is released under the [MIT License](LICENSE). The China-Fundus-CIMT dataset is distributed separately by its original authors and is subject to its own licensing and usage conditions.

## Disclaimer

This repository is intended for **research and academic purposes only**. The models and code are not intended for clinical diagnosis or clinical decision-making.
