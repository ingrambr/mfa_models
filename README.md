# MFA Prediction Models

This repository contains a collection of machine learning models for predicting Microfibril Angle (MFA) from X-ray diffraction profiles. The code implements and evaluates Random Forest, k-Nearest Neighbors, and Artificial Neural Network approaches.

## Data Files

The models use two main data files that can be downloaded from:
https://doi.org/10.6084/m9.figshare.28458716
- profiles-20250221.csv - Contains the input X-ray diffraction profiles
- params-20250221.csv - Contains the target MFA values

## Test Profiles

All models are evaluated against six consistent real-world test profiles, files that are included in this repository:
- interpolada185.txt - Diffraction profile from sample 185
- interpolada230.txt - Diffraction profile from sample 230  
- interpolada303.txt - Diffraction profile from sample 303
- earlywood.csv - Typical earlywood profile
- latewood.csv - Typical latewood profile
- compressionwood.csv - Typical compression wood profile

## Model Implementation Files

### Random Forest Models
- `mfa-RF-testsets-split.py` - Analyzes how different train/test split ratios affect Random Forest model performance
- `mfa-RF-generate-model.py` - Trains and evaluates Random Forest models for MFA prediction using optimal parameters

### k-Nearest Neighbors Models  
- `mfa-Knn-testsets-split.py` - Analyzes how different train/test split ratios affect kNN model performance
- `mfa-Knn-generate-model.py` - Implements kNN model training and evaluation with optimized parameters

### Artificial Neural Network Models
- `mfa-ANN-testsets-split.py` - Analyzes how different train/test split ratios affect ANN model performance
- `mfa-ANN-generate-model.py` - Trains and evaluates ANN models using the optimal architecture and parameters
- `mfa-ANN-layers.py` - Experiments with different neural network architectures to optimize layer configuration

Each implementation includes:
- Data preprocessing and normalization
- Model training and validation
- Performance evaluation using multiple metrics (ME, MAE, RMSE, R)
- Prediction testing on the six reference profiles
- Support for both MFA and desMFA prediction

## Dependencies

- numpy
- pandas
- scikit-learn
- pytorch (for ANN models)
  
