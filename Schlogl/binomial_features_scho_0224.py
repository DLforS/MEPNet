#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:24:33 2024

@author: yangwuyue
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
from scipy.special import comb


def binomial_features_time(x, t_values, degree=10):
    all_features = []
    for t in t_values:
        features_t = [comb(degree, k) * (x ** k) * ((1 - x) ** (degree - k)) for k in range(degree + 1)]
        features_t = np.array(features_t).T  
        time_c = np.full((features_t.shape[0], 1), t)
        features_time = np.hstack([features_t, time_c])  
        all_features.append(features_time)
    features = np.stack(all_features, axis=1)  
    return torch.tensor(features, dtype=torch.float32)

class MEPNet(nn.Module):
    def __init__(self, input_dim, time_dim):
        super(MEPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, time_dim, bias=False)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.softplus(self.fc3(x))

def cal_entropy(px):
    epsilon = 1e-10
    entropy = torch.sum(px * torch.log(px + epsilon))
    return entropy

def loss_func(model, moments, x, true_px):
    px = model(x).squeeze()  # [100*3]
    entropy = cal_entropy(px)
    moment_constraints = torch.sum(px[:,:, None] * x[:,:,0:-1], dim=0) - moments
    mse = torch.mean((px - true_px.squeeze())**2)
    loss = torch.mean(moment_constraints**2)
    return loss, mse, entropy, px


import pandas as pd
data = pd.read_csv('histogram_values_1e4-4的副本.csv', header=None)
x_values = data.iloc[0, :].values
histogram = data.iloc[1:, :].values.astype(float)
histogram = np.array(histogram).T

#t_values = np.array([0.01,0.05,0.1,0.5,1,2,3,4,5])  
t_values = np.linspace(0.1, 5, 50)  # 0.1:0.1:5
x_values = np.linspace(0, 1, histogram.shape[0])#50 725

input_tensor = binomial_features_time(x_values, t_values)
gauss_tensor = torch.tensor(histogram, dtype=torch.float32).unsqueeze(-1) #135,9,1

moments = []
for i in range(input_tensor.shape[2]-1):
    moment = torch.sum(gauss_tensor * input_tensor[:,:, i:i+1], dim=0)
    moments.append(moment)
moments = torch.cat(moments, dim=1)


time_dimension = t_values.shape[0]
feature_dimension = 10 + 1 + 1
input_dim = time_dimension * feature_dimension

model = MEPNet(input_dim, time_dimension)
optimizer = Adam(model.parameters(), lr=0.001)

loss_history = []
mse_history = []
entropy_history = []
min_mse = float('inf')
for step in range(100000):  
    optimizer.zero_grad()
    loss, mse, entropy, px = loss_func(model, moments, input_tensor, gauss_tensor)
    mse.backward()
    optimizer.step()

    loss_history.append(loss.item())
    mse_history.append(mse.item())
    entropy_history.append(entropy)

    if mse.item() < min_mse:
        min_mse = mse.item()
        best_model_params = model.state_dict().copy()

    if (step + 1) % 1000 == 0:
        print(f'Step {step+1}, Loss: {loss.item()}, MSE: {mse.item()}')


with open("training_records_1e4.txt", "w") as file:
    file.write("Step\tLoss\tMSE\tEntropy\n")
    for step in range(len(loss_history)):
        file.write(f"{step+1}\t{loss_history[step]}\t{mse_history[step]}\t{entropy_history[step]}\n")

def model_predictions(model, x, t_values):
    input_tensor = binomial_features_time(x, t_values)
    model.eval()  
    with torch.no_grad():
        predictions = model(input_tensor)  
    predictions = predictions.view(len(x), len(t_values)).numpy()  
    return predictions

model.load_state_dict(best_model_params)
model_pred = model_predictions(model, x_values, t_values)

plt.figure(figsize=(6, 3))
X, T = np.meshgrid(x_values, t_values)
plt.subplot(1, 3, 1)
plt.contourf(X, T, histogram.T, 20, cmap='RdGy')
plt.colorbar()
plt.title('True')
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.subplot(1, 3, 2)
plt.contourf(X, T, model_pred.T, 20, cmap='RdGy')
plt.colorbar()
plt.title('MEP-Net (10 order)')
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.subplot(1, 3, 3)
plt.contourf(X, T, model_pred.T, 20, cmap='RdGy')
plt.colorbar()
plt.title('Error')
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.tight_layout()
plt.show()
#t_values = np.linspace(0.1,5,50)
#t_values = np.array([0.01,0.05,0.1,0.5,1,2,5])  
def plot_predictions(model, x, t_values, true, num=5):
    plt.figure(figsize=(8, 6))
    model_pred = model_predictions(model, x, t_values)
    #selected_indices = np.linspace(0, len(t_values) - 1, num_points_to_plot, dtype=int)
    for idx, t in enumerate(t_values):
        #plt.subplot(3, 3, idx + 1)
        true_values = true[:, idx]
        plt.plot(x, true_values, label='Hist', color='blue')
        predictions_at_t = model_pred[:, idx]
        plt.plot(x, predictions_at_t, '--', label='MEP-Net', color='red')
        
        plt.title(f'T = {t:.2f}')
        plt.xlabel('$x$')
        plt.ylabel('$p(x)$')
        #plt.legend()
    plt.tight_layout()
    plt.show()

plot_predictions(model, x_values, t_values, histogram, num=3)
