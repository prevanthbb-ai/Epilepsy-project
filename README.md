# Automated Seizure Detection Using CNN–BiLSTM with Attention
### Deep Learning for Fast and Accessible Epilepsy Diagnosis
**Author:** Prathapani Revanth  
**Dataset:** CHB-MIT Scalp EEG Database  
**Model:** CNN + Bi-LSTM + Custom Attention Layer  

---

## Overview
This project presents an automated seizure detection system using a deep learning hybrid architecture that combines **1D CNN**, **Bi-LSTM**, and a **custom attention mechanism**. The model is designed to classify seizure vs. non-seizure EEG segments quickly and accurately, with the goal of improving access to diagnostic tools in low-resource clinical settings.

---

## Key Features
- CNN layers extract spatial EEG patterns  
- Bi-LSTM captures long-term temporal structure  
- Attention mechanism highlights seizure-relevant time steps  
- Gaussian noise + synthetic seizures for data augmentation  
- Trained on the CHB-MIT Scalp EEG dataset  
- Strong performance with a low false-negative rate  

---

## Results
| Metric | Score |
|--------|--------|
| **Accuracy** | 93.7% |
| **Specificity** | 94% |
| **Sensitivity** | 73.1% |

Figures (confusion matrix, training curves, spectrograms) are provided in the **figures/** folder.

---

