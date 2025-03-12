#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Random Forest Model Generator for MFA Prediction

This script trains and evaluates Random Forest models for MFA and desMFA prediction.

Download datafiles (profiles-20250221.csv & params-20250221.csv)
from: https://doi.org/10.6084/m9.figshare.28458716

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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

print("Variable Template n_trees ME MAE RMSE R2 P1 P2 P3 EW LW CW")

# Model parameter - number of trees in the forest
n_trees = 50

# Train for each template (0 = all templates)
for whichTempl in [0, 1, 2, 3]:
    # Filter data by template if specified
    if whichTempl > 0:
        dft_filtered = dft[dft["template"] == whichTempl].copy(deep=True)
        dfd_filtered = dfd[dft["template"] == whichTempl]
    else:
        dft_filtered = dft.copy(deep=True)
        dfd_filtered = dfd

    # Normalize the feature data
    X_data = dfd_filtered
    X_data = (X_data / np.tile(X_data.sum(1), [360, 1]).T)

    # Train for each prediction column
    for predCol in ["MFA", "desMFA"]:
        # Load test patterns
        X_test_patterns = load_test_patterns(predCol)

        # Prepare target data
        y_data = np.abs(dft_filtered[predCol].values.reshape(-1, 1))

        # Create train/test split with indices
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X_data, y_data, dfd_filtered.index, test_size=0.5, random_state=42
        )

        # Train Random Forest model
        rf = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        rf.fit(X_train, y_train.ravel())  # Using ravel() to convert to 1D array

        # Make predictions
        y_pred = rf.predict(X_test).reshape(-1, 1)
        y_pred_train = rf.predict(X_train).reshape(-1, 1)
        test_pattern_preds = rf.predict(X_test_patterns).reshape(-1, 1)

        # Calculate metrics
        me = np.mean(y_pred - y_test)  # Mean Error
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        correlation = np.corrcoef(y_test.ravel(), y_pred.ravel())[0, 1]

        # Print results
        output_line = "{:s} {:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}".format(
            predCol, whichTempl, n_trees, me, mae, rmse, r2
        )

        # Add test pattern predictions to output
        for tp_pred in test_pattern_preds:
            output_line += " {:.3f}".format(tp_pred[0])

        print(output_line, flush=True)

