# Video Summarization using CNN–LSTM (SumMe Dataset)

## Overview

This project implements an end-to-end video summarization system using deep learning.  
The goal is to automatically generate a short summary video from a longer input video by identifying the most important frames based on learned temporal patterns.

The system is trained using human-annotated importance scores from the **SumMe dataset** and can generate summaries for unseen videos **without any ground-truth annotations**.

---

## Key Features

- CNN-based feature extraction using **ResNet-18**
- Temporal modeling using **LSTM**
- Multi-video training for better generalization
- Supports summarization of random unseen videos
- No ground-truth required during inference
- Clean and modular code structure

---

## Problem Statement

Watching long videos is time-consuming.  
This project aims to automatically create concise summaries that preserve the most important content while maintaining temporal coherence and diversity.

---

## System Pipeline

→ Frame Sampling (2 FPS)
→ CNN Feature Extraction
→ LSTM Sequence Modeling
→ Frame Importance Prediction
→ Frame Selection
→ Summary Video
---

## Project Structure

Video-Summarization-CNN-LSTM/
├── Code/
│ ├── extract_features.py
│ ├── dataset_multi.py
│ ├── model.py
│ ├── train_all_videos.py
│ ├── inference.py
│ └── random_video_summ.py
│
├── README.md
├── requirements.txt
└── .gitignore
---

## Dataset

**SumMe Video Summarization Dataset**

- Contains short consumer videos
- Each frame annotated with human importance scores
- Used only during training
- Dataset is not included in this repository due to size constraints

Ground-truth annotations are never used during inference.

---

## Code Explanation

### extract_features.py
Extracts visual features from videos using a pretrained ResNet-18 model.  
Frames are sampled at 2 FPS and converted into 512-dimensional feature vectors.

---

### dataset_multi.py
Loads features and ground-truth scores from all videos, aligns them, and generates temporal sequences for training.

---

### model.py
Defines an LSTM-based neural network that predicts frame importance based on temporal context.

---

### train_all_videos.py
Trains a single LSTM model using all available videos to improve generalization and avoid overfitting.

---

### inference.py
Generates summary videos for dataset videos using the trained model without retraining.

---

### random_video_summ.py
End-to-end summarization of any random unseen video.  
This script performs frame extraction, feature extraction, importance prediction, and summary video generation without using any ground-truth scores.

---

## Evaluation

### Qualitative Evaluation
- Visual inspection of generated summaries
- Coverage of key events
- Temporal coherence
- Frame diversity

### Quantitative Evaluation (Optional)
- Can be performed on the SumMe dataset using F-score
- Requires ground-truth annotations
- Not applicable for random unseen videos

---

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt

