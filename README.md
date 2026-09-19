# Credit Card Fraud Detection using TensorFlow

A supervised deep learning model built from scratch to classify credit card transactions as legitimate or fraudulent. This project transitions from an unsupervised anomaly detection concept into a fully supervised binary classification pipeline.

## Project Structure
- **`Supervised_Training_Set.xlsx`**: Training dataset containing mixed normal and fraudulent transaction samples.
- **`Supervised_CV_Set.xlsx`**: Cross-validation dataset used for unbiased model evaluation.
- **`fraud_detector.py`**: The complete end-to-end Python script covering data loading, feature selection, standardization, neural network architecture, training, and loss tracking.

## Features Used
The model trains exclusively on behavioral and numerical attributes to prevent ID memorization overfitting:
- Transaction Amount (`amt`)
- Geographical Coordinates (`lat`, `long`)
- City Population (`city_pop`)
- Merchant Coordinates (`merch_lat`, `merch_long`)

## Model Architecture
Built using TensorFlow/Keras Sequential API:
1. **Input Layer**: 6 numerical features.
2. **Hidden Layer 1**: 16 neurons with `ReLU` activation.
3. **Hidden Layer 2**: 8 neurons with `ReLU` activation.
4. **Output Layer**: 1 neuron with `Sigmoid` activation (outputs probability of fraud).

## Installation & Requirements
Make sure you have the required libraries installed before running the script:
```bash
pip install tensorflow pandas scikit-learn openpyxl matplotlib
