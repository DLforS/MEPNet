#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:11:48 2024

@author: yangwuyue
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the min_max_values_1e4_jiange5.csv file
min_max_values_path = 'min_max_values_1e6_jiange5.csv'
min_max_values_df = pd.read_csv(min_max_values_path)

# Load and parse the all_histogram_values_1e4_jiange5.csv file
histogram_values_path = 'all_histogram_values_1e6_jiange5.csv'
histogram_values_df = pd.read_csv(histogram_values_path, header=None)
histogram_values_list = histogram_values_df[0].str.split(expand=True).apply(pd.to_numeric)

# Load the model_predictions.txt file
predictions_path = 'model_predictions.txt'
predictions_df = pd.read_csv(predictions_path, header=None, sep='\t')

# Prepare the X and Y coordinates
x_values = np.linspace(min_max_values_df.iloc[:, 0].min(), min_max_values_df.iloc[:, 1].max(), histogram_values_list.shape[1])

# Adjust the Y values to match the Z shape
Z = histogram_values_list.to_numpy()
adjusted_y_values = np.linspace(0.1, 5, 50)

# Create a meshgrid with the adjusted Y values
X_adj, Y_adj = np.meshgrid(x_values, adjusted_y_values)

# Plotting the contour map with the adjusted meshgrid
plt.figure(figsize=(10, 6))
contour_adj = plt.contourf(X_adj, Y_adj, Z, levels=30, cmap='RdGy')
plt.colorbar(contour_adj)
plt.title('Adjusted Contour Plot of Histogram Values Over Time')
plt.xlabel('X Values')
plt.ylabel('Time Intervals (Adjusted)')
plt.show()

# Selecting key moments for plotting
# Assuming the key moments are at the beginning, middle, and end of the time intervals
key_moments_indices = [0, Z.shape[0] // 2, Z.shape[0] - 1]

# Plotting distributions for the selected key moments
plt.figure(figsize=(15, 5))

for i, idx in enumerate(key_moments_indices, 1):
    plt.subplot(1, len(key_moments_indices), i)
    plt.plot(x_values, Z[idx, :], label=f"Real Values at Time: {adjusted_y_values[idx]:.1f}")
    plt.plot(x_values, predictions_df.iloc[idx, :], label=f"Predicted Values at Time: {adjusted_y_values[idx]:.1f}")
    plt.xlabel('X Values')
    plt.ylabel('Frequency')
    plt.title(f"Distribution at Time {adjusted_y_values[idx]:.1f}")
    plt.legend()

plt.tight_layout()
plt.show()