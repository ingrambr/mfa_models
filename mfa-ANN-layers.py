#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ANN Architecture Selection Experiment for layer arrangement

This script evaluates how different layers numbers and neurons affect model performance.

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

class ANNModel(nn.Module):
    def __init__(self, layer_sizes):
        super(ANNModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all but the last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)
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

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, X_test_patterns,
                             epochs=5000, learning_rate=0.001, early_stopping=True):
    """Train the model and evaluate on test data"""

    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test)
    X_test_tensor_tp = torch.FloatTensor(X_test_patterns)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    best_test_rmse = float('inf')
    patience = 10
    counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Test evaluation step
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)
                test_rmse = np.sqrt(test_loss.item())

                # Check if test performance improved
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    counter += 1

                # Early stopping
                if early_stopping and counter >= patience:
                    break

    # Load best model if applicable
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor).numpy()
        test_pattern_preds = model(X_test_tensor_tp).numpy()

        # Calculate metrics
        mae = mean_absolute_error(y_test, test_outputs)
        rmse = np.sqrt(mean_squared_error(y_test, test_outputs))
        r2 = r2_score(y_test, test_outputs)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'test_pattern_preds': test_pattern_preds
    }

def get_architecture_configs():
    """Generate architecture configurations with preference for rounded neuron counts"""

    configs = [
        [360, 90, 1],
        [360, 180, 1],
        [360, 120, 1],

        [360, 90, 30, 1],
        [360, 180, 60, 1],
        [360, 120, 40, 1],

        [360, 90, 30, 10, 1],
        [360, 180, 90, 30, 1],
        [360, 120, 60, 20, 1],

        [360, 180, 90, 30, 10, 1],
        [360, 150, 75, 35, 15, 1],
        [360, 90, 30, 10, 5, 1]

    ]
    return configs

predCol = "MFA"

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
y_data = np.abs(dft[predCol].values.reshape(-1, 1))

# Create a 50:50 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.5, random_state=42
)

# Load test patterns
X_test_patterns = load_test_patterns(predCol)

# Architecture parameters
epochs = 15000
learning_rate = 0.0001

print("Architecture, Num_Layers, RMSE, MAE, R2, P1, P2, P3, EW, LW, CW")

# Store results for plotting
results = []

# Get predefined configurations
configs = get_architecture_configs()


# Test each architecture configuration
for layer_sizes in configs:
    # Number of layers (including input and output)
    num_layers = len(layer_sizes)

    # Report layer structure
    layer_str = "→".join([str(size) for size in layer_sizes])

    # Create and train model
    model = ANNModel(layer_sizes)
    metrics = train_and_evaluate_model(
        model, X_train, y_train, X_test, y_test,
        X_test_patterns, epochs, learning_rate
    )

    # Store results
    results.append({
        'layer_sizes': layer_sizes,
        'num_layers': num_layers,
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'r2': metrics['r2'],
        'test_pattern_preds': metrics['test_pattern_preds']
    })

    # Display test pattern predictions
    test_pattern_str = ", ".join([f"{pred[0]:.3f}" for pred in metrics['test_pattern_preds']])

    print(f"{layer_str}, {num_layers}, {metrics['rmse']:.4f}, {metrics['mae']:.4f}, {metrics['r2']:.4f}, {test_pattern_str}")


