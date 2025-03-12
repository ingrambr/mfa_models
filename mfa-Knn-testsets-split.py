#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train/Test Split Ratio Analysis for MFA Prediction

This script evaluates how different train/test split ratios impact Knn model performance.

Download datafiles (profiles-20250221.csv & params-20250221.csv)
from: https://doi.org/10.6084/m9.figshare.28458716

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

# specific real world test patterns
tp1fName = "interpolada185.txt"
tp2fName = "interpolada230.txt"
tp3fName = "interpolada303.txt"
tp4fName = "earlywood.csv"
tp5fName = "latewood.csv"
tp6fName = "compressionwood.csv"

fNameData = "profiles-20250221.csv"
fNameTargets = "params-20250221.csv"
dfd = pd.read_csv(fNameData, index_col="id")
dft = pd.read_csv(fNameTargets)

# Load test patterns
def load_test_patterns(predCol):
    dftp1 = pd.read_csv(tp1fName, delimiter=" ", index_col=0, header=None, names=[predCol])
    dftp2 = pd.read_csv(tp2fName, delimiter=" ", index_col=0, header=None, names=[predCol])
    dftp3 = pd.read_csv(tp3fName, delimiter=" ", index_col=0, header=None, names=[predCol])
    dftp4 = pd.read_csv(tp4fName, delimiter=",", index_col=0, header=None, names=[predCol])
    dftp5 = pd.read_csv(tp5fName, delimiter=",", index_col=0, header=None, names=[predCol])
    dftp6 = pd.read_csv(tp6fName, delimiter=",", index_col=0, header=None, names=[predCol])

    X_data_tp = np.vstack([dftp1.values.T, dftp2.values.T, dftp3.values.T, dftp4.values.T, dftp5.values.T, dftp6.values.T])
    X_data_TP = (X_data_tp / np.tile(X_data_tp.sum(1), [360, 1]).T)

    return X_data_TP

print("Variable Template TrainRatio Epochs ME MAE RMSE R2 P1 P2 P3 EW LW CW")

# Set which template to use (0 = all templates)
whichTempl = 0

# Filter data by template if specified
if whichTempl > 0:
    dft = dft[dft["template"] == whichTempl].copy(deep=True)
    dfd = dfd[dft["template"] == whichTempl]
else:
    dft = dft.copy(deep=True)
    dfd = dfd

# Normalize the feature data
X_data = dfd
X_data = (X_data / np.tile(X_data.sum(1), [360, 1]).T)

for predCol in ["MFA", "desMFA"]:
    # Load test patterns
    X_test_patterns = load_test_patterns(predCol)

    y_data = np.abs(dft[predCol].values.reshape(-1, 1))

    # Define the neighbor parameter
    neighbors = 11

    # Iterate through different train/test split ratios
    for train_ratio in np.arange(0.05, 0.96, 0.05):
        # Round to 2 decimal places for cleaner output
        train_ratio = round(train_ratio, 2)

        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, train_size=train_ratio, random_state=42
        )

        # Train kNN model
        knn = KNeighborsRegressor(n_neighbors=neighbors)
        knn.fit(X_train, y_train.ravel())

        # Make predictions
        y_pred = knn.predict(X_test).reshape(-1, 1)
        test_pattern_preds = knn.predict(X_test_patterns).reshape(-1, 1)

        # Calculate metrics
        me = np.mean(y_pred - y_test)  # Mean Error
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Print results
        output_line = "{:s} {:d} {:.2f} {:d} {:.3f} {:.3f} {:.3f} {:.3f}".format(
            predCol, whichTempl, train_ratio, neighbors, me, mae, rmse, r2
        )

        # Add test pattern predictions to output
        for tp_pred in test_pattern_preds:
            output_line += " {:.3f}".format(tp_pred[0])

        print(output_line, flush=True)


