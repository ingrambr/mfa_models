#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ANN Train/Test Split Ratio Analysis for MFA Prediction

This script evaluates how different train/test split ratios affect model performance.

Download datafiles (profiles-20250221.csv & params-20250221.csv)
from: https://doi.org/10.6084/m9.figshare.28458716

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ANN(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3, hs4):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hs1)
        self.fc2 = nn.Linear(hs1, hs2)
        self.fc3 = nn.Linear(hs2, hs3)
        self.fc4 = nn.Linear(hs3, hs4)
        self.fc5 = nn.Linear(hs4, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x

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

# Architecture parameters - Funnel architecture with progressively fewer neurons
input_size = 360
hidden_size1 = 90   # Reduced from input size by ~75%
hidden_size2 = 30   # Reduced from previous layer by ~67%
hidden_size3 = 10   # Reduced from previous layer by ~67%
hidden_size4 = 5    # Reduced from previous layer by 50%
epochs = 15000      # Reduced for demonstration purposes
learning_rate = 0.0001

for predCol in ["MFA", "desMFA"]:
    # Load test patterns
    X_test_patterns = load_test_patterns(predCol)
    X_test_tensor_tp = torch.FloatTensor(X_test_patterns)

    y_data = np.abs(dft[predCol].values.reshape(-1, 1))

    # Iterate through different train/test split ratios
    for train_ratio in np.arange(0.05, 0.96, 0.05):
        # Round to 2 decimal places for cleaner output
        train_ratio = round(train_ratio, 2)

        # Create train/test split
        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
            X_data, y_data, train_size=train_ratio, random_state=42
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_data.values)
        y_train_tensor = torch.FloatTensor(y_train_data)
        X_test_tensor = torch.FloatTensor(X_test_data.values)
        y_test_tensor = torch.FloatTensor(y_test_data)

        # Create and train the model
        model = ANN(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate model
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy()
            test_pattern_preds = model(X_test_tensor_tp).numpy()

            # Calculate metrics
            me = np.mean(y_pred - y_test_data)  # Mean Error
            mae = mean_absolute_error(y_test_data, y_pred)
            mse = mean_squared_error(y_test_data, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_data, y_pred)

            # Print results
            output_line = "{:s} {:d} {:.2f} {:d} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                predCol, whichTempl, train_ratio, epochs, me, mae, rmse, r2
            )

            # Add test pattern predictions to output
            for tp_pred in test_pattern_preds:
                output_line += " {:.3f}".format(tp_pred[0])

            print(output_line, flush=True)
