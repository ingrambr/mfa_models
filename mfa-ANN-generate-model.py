#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ANN Model Generator for MFA Prediction

This script trains and evaluates an ANN model to predict MFA and desMFA.

Download datafiles (profiles-20250221.csv & params-20250221.csv)
from: https://doi.org/10.6084/m9.figshare.28458716

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Function to load test patterns
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

print("Variable Template Epoch ME MAE RMSE R P1 P2 P3 EW LW CW")

# Model architecture parameters
input_size = 360
hidden_size1 = 90
hidden_size2 = 30
hidden_size3 = 10
hidden_size4 = 5
epochs = 15000
eval_steps = 1000
learning_rate = 0.0001

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
        X_test_tensor_tp = torch.FloatTensor(X_test_patterns)

        # Prepare target data
        y_data = np.abs(dft_filtered[predCol].values.reshape(-1, 1))

        # Create train/test split with indices
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X_data, y_data, dfd_filtered.index, test_size=0.5, random_state=42
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.FloatTensor(y_test)

        # Create and train the model
        model = ANN(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop with periodic evaluation
        for epoch in range(epochs):
            # Regular training step
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Periodic evaluation
            if (epoch % eval_steps) == (eval_steps - 1):
                # Intensive training phase (as in the original script)
                for epoch2 in range(int(eval_steps / 10)):
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()

                # Evaluate model
                with torch.no_grad():
                    y_pred = model(X_test_tensor).numpy()
                    y_pred_train = model(X_train_tensor).numpy()
                    test_pattern_preds = model(X_test_tensor_tp).numpy()

                    # Calculate metrics
                    me = np.mean(y_pred - y_test)  # Mean Error
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    correlation = np.corrcoef(y_test.reshape(-1), y_pred.reshape(-1))[0, 1]

                    # Print results
                    output_line = "{:s} {:d} {:d} {:7.3f} {:7.3f} {:7.3f} {:7.3f}".format(
                        predCol, whichTempl, epoch, me, mae, rmse, correlation
                    )

                    # Add test pattern predictions to output
                    for tp_pred in test_pattern_preds:
                        output_line += " {:7.3f}".format(tp_pred[0])

                    print(output_line, flush=True)


        print("Final model:")
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy()
            y_pred_train = model(X_train_tensor).numpy()
            test_pattern_preds = model(X_test_tensor_tp).numpy()

            # Calculate metrics
            me = np.mean(y_pred - y_test)  # Mean Error
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            correlation = np.corrcoef(y_test.reshape(-1), y_pred.reshape(-1))[0, 1]

            # Print results
            output_line = "{:s} {:d} {:d} {:7.3f} {:7.3f} {:7.3f} {:7.3f}".format(
                predCol, whichTempl, epoch, me, mae, rmse, correlation
            )

            # Add test pattern predictions to output
            for tp_pred in test_pattern_preds:
                output_line += " {:7.3f}".format(tp_pred[0])

            print(output_line, flush=True)
            
            
            
