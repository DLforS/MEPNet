# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:42:23 2023

@author: DELL
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
from scipy.special import comb



def custom_activation(x, alpha=1, beta=1):
    return alpha * torch.exp(x-1) + beta * torch.exp(x*x)

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetworkModel, self).__init__()

        self.fc1 = nn.Linear(input_dim,50)  
        self.fc2 = nn.Linear(50, 50) 
        self.fc3 = nn.Linear(50, 1, bias=False)          

    def forward(self, x):
        x =  torch.tanh(self.fc1(x))  
        x = self.fc2(x) 
        x = self.fc3(x)
        return F.softplus(x)

def objective_function(model, moments, x, true_px, entropy_weight, prev_px):

    px = model(x).squeeze()  
    entropy = torch.sum(px * torch.log(px / (prev_px + 1e-10)))
    moment_constraints = torch.sum(px[:, None] * x, dim=0) - moments
    moment_loss = torch.mean(moment_constraints**2)
    #print(moment_loss,entropy)
    loss = moment_loss + 1e-12 * entropy
    mse = torch.mean((px - true_px.squeeze())**2)
    return loss, px, moment_loss, entropy, mse


# 高斯分布
def gauss3(x):
    σ0 = 1.0/14.0
    μ0 = 1.0/4.0
    A0 = 1.0/(2.0*σ0*np.sqrt(2*np.pi))
    σ1 = 1.0/20.0
    μ1 = 2.0/4.0
    A1 = 1/(2*σ1*np.sqrt(2*np.pi))
    return A0*np.exp(-(x-μ0)**2/(2*σ0**2)) + A1*np.exp(-(x-μ1)**2/(2*σ1**2))

x = np.linspace(0, 1, 100)

x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
gauss_tensor = torch.tensor(gauss3(x), dtype=torch.float32).view(-1, 1)

n = 30  
binomial_features = [comb(n, k) * (x_tensor ** k) * ((1 - x_tensor) ** (n - k)) for k in range(n + 1)]

x_tensor = torch.cat(binomial_features, dim=1)              

moments = []
for i in range(x_tensor.shape[1]):
    moment = torch.sum(gauss_tensor * x_tensor[:, i:i+1], dim=0)
    moments.append(moment)
moments = torch.cat(moments, dim=0)

model = NeuralNetworkModel(input_dim=x_tensor.shape[1])

optimizer = Adam(model.parameters(), lr=0.001)

lambda_history = []
entropy_history = []
loss_history = []
mse_history = []
min_mse = float('inf')
best_model_params = None

def adjust_entropy_weight(initial_entropy_weight, current_iteration, max_iterations):
    alpha = 0.1  
    return initial_entropy_weight * (alpha ** (current_iteration / max_iterations))

initial_entropy_weight = 1e-6 
max_iterations = 10000000  

prev_px = torch.ones_like(gauss_tensor)

for step in range(max_iterations):  
    entropy_weight = adjust_entropy_weight(initial_entropy_weight, step, max_iterations) 
    optimizer.zero_grad()
    loss, px, moment_loss, entropy, mse = objective_function(model, moments, x_tensor, gauss_tensor, entropy_weight, prev_px)
    loss.backward()
    optimizer.step()
    prev_px = px.detach().clone()
    
    if mse.item() < min_mse:
        min_mse = mse.item()
        best_model_params = model.state_dict().copy()  

    if (step + 1) % 100000 == 0:
        print(f'Step {step+1}, Loss: {loss.item()}, Entropy: {entropy.item()}, moment_loss: {moment_loss.item()}, mse: {mse.item()}')

    entropy_history.append(entropy.item())
    loss_history.append(loss.item())
    mse_history.append(mse.item())

model.load_state_dict(best_model_params)
px = model(x_tensor)
plt.plot(x, gauss3(x), 'b--', label='True')
plt.plot(x, px.detach().numpy(), 'r-', label='MEP-Net')  
plt.show()

with open('history_moment_entropy.txt', 'w') as file:
    for i in range(len(entropy_history)):
        file.write(f'{entropy_history[i]}, {loss_history[i]}, {mse_history[i]}\n')

fig, axs = plt.subplots(1, 3, figsize=(20, 6))

axs[0].plot(x, gauss3(x), 'b--', label='True')
axs[0].plot(x, px.detach().numpy(), 'r-', label='MaxEn-Net')
axs[0].set_xlabel('x')
axs[0].set_ylabel('p(x)')
axs[0].legend()

axs[1].plot(entropy_history, 'm-')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Entropy')

axs[2].plot(np.log(mse_history), 'g-')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Loss')

plt.tight_layout()
plt.show()

px = px.detach().numpy()  
axs[2, 0].plot(x, gauss3(x), label='True Distribution')
axs[2, 0].plot(x, px, label='MaxEnt Reconstruction')
axs[2, 0].set_title('Probability Distribution')
axs[2, 0].legend()

plt.show()

model.load_state_dict(best_model_params)
px = model(x_tensor)
final_px = px.detach().numpy()

np.savetxt('final_px_30order_entropy.txt', final_px, delimiter=',')

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import comb

x = np.linspace(0, 1, 100)
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)

n = 10  
binomial_features = [comb(n, k) * (x_tensor ** k) * ((1 - x_tensor) ** (n - k)) for k in range(n + 1)]

plt.figure(figsize=(6, 4))
for i, feature in enumerate(binomial_features):
    plt.plot(x, feature.numpy(), label=f'Feature {i+1}')
    plt.xlabel('$x$')
    plt.ylabel('Feature')

plt.tight_layout()
plt.show()
