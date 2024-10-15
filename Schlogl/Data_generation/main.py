# -*- coding: utf-8 -*-
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load the data
data = scipy.io.loadmat('Y_samples_data.mat')
Y_samples = data['Y_samples']

# Extract the data for x, a, b across all runs and time points
x_data = Y_samples[0, :, :]
a_data = Y_samples[1, :, :]
b_data = Y_samples[2, :, :]
'''
# Convert to PyTorch tensors
x_tensor = torch.tensor(x_data, dtype=torch.float32, requires_grad = True)
a_tensor = torch.tensor(a_data, dtype=torch.float32, requires_grad = True)
b_tensor = torch.tensor(b_data, dtype=torch.float32, requires_grad = True)

# Prepare the time variable
num_time_points = Y_samples.shape[2]
time = np.linspace(0, 10, num_time_points)  # Replace with actual time range
t_tensor = torch.tensor(time, dtype=torch.float32, requires_grad = True)
t_tensor_org = torch.tensor(time, dtype=torch.float32)
t_tensor = t_tensor.repeat(Y_samples.shape[1], 1)

# Combine x, a, b, and t for the input data
input_data = torch.stack((x_tensor, a_tensor, b_tensor, t_tensor), dim=2)
'''
num_time_points = Y_samples.shape[2]
time = np.linspace(0, 10, num_time_points)  # Replace with actual time range
x_flat = x_data.flatten()
a_flat = a_data.flatten()
b_flat = b_data.flatten()
t_flat = np.tile(time, (x_data.shape[0], 1)).flatten()
input_data = np.column_stack([x_flat, a_flat, b_flat, t_flat])
input_tensor = torch.tensor(input_data, dtype=torch.float32)
x_tensor = torch.tensor(x_flat, dtype=torch.float32)
a_tensor = torch.tensor(a_flat, dtype=torch.float32)
b_tensor = torch.tensor(b_flat, dtype=torch.float32)
t_tensor = torch.tensor(t_flat, dtype=torch.float32)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(4, 50)  # 4 input features: x, a, b, t
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)  # 3 outputs: predictions for x, a, b
        self.k1 = nn.Parameter(torch.tensor([0.0015], requires_grad=True))  # Example initial values
        self.k2 = nn.Parameter(torch.tensor([0.15], requires_grad=True))
        self.k3 = nn.Parameter(torch.tensor([20.0], requires_grad=True))
        self.k4 = nn.Parameter(torch.tensor([3.5], requires_grad=True))
        #k_values = [0.0015, 0.15, 20, 3.5]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
    
class LambdaNetwork(nn.Module):
    def __init__(self):
        super(LambdaNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input is time t
        self.fc2 = nn.Linear(10, 6)  # Output 6 lambda values (3 for 1st order, 3 for 2nd order)

    def forward(self, t):
        t = torch.relu(self.fc1(t))
        lambda_values = torch.exp(self.fc2(t))  # Use exp to ensure positivity
        return lambda_values


# Define loss functions
def max_entropy_loss(p):
    return -torch.sum(p * torch.log(p + 1e-6))

def moment_loss(predicted, actual):
    return torch.mean((predicted - actual) ** 2)

def calculate_moments(p, values, moment_order):
    moment = torch.sum(p * values ** moment_order, dim=0)
    return moment

def calculate_moment(data_tensor, moment_order):
    """计算给定数据的指定阶矩"""
    return torch.mean(data_tensor ** moment_order, dim=0)

def physics_loss(p_pred, input_data, model):
    # 从 input_data 中提取 x, a, b 和 t
    num_samples = Y_samples.shape[1]  # Assuming Y_samples is accessible here
    num_time_points = Y_samples.shape[2]

    # Reshape p_pred to match the original data structure
    p_pred1 = p_pred.view(num_samples, num_time_points)

    # Extract x, a, b, and t from input_data
    x = input_data[:, 0].view(num_samples, num_time_points)
    a = input_data[:, 1].view(num_samples, num_time_points)
    b = input_data[:, 2].view(num_samples, num_time_points)
    t = input_data[:, 3].view(num_samples, num_time_points)

    '''
    x = input_data[:, 0].requires_grad_(True)
    a = input_data[:, 1].requires_grad_(True)
    b = input_data[:, 2].requires_grad_(True)
    t = input_data[:, 3].requires_grad_(True)
    '''
    # 计算加权平均值
    avg_x = torch.sum(p_pred1 * x, dim=0)
    avg_a = torch.sum(p_pred1 * a, dim=0)
    avg_b = torch.sum(p_pred1 * b, dim=0)
    
    # 为了计算梯度，需要确保 avg_x, avg_a, avg_b 支持梯度
    #avg_x.requires_grad_(True)
    #avg_a.requires_grad_(True)
    #avg_b.requires_grad_(True)
    #t_tensor.requires_grad_(True)
    avg_x = torch.tile(avg_x, (1000, 1)).flatten()
    avg_a = torch.tile(avg_a, (1000, 1)).flatten()
    avg_b = torch.tile(avg_b, (1000, 1)).flatten()
    dummy = torch.ones_like(p_pred)
    actual_dx = torch.autograd.grad(outputs=avg_x, inputs=input_data, grad_outputs=dummy, create_graph=True)[0]
    actual_dx_dt = actual_dx[:, 3]
    actual_da = torch.autograd.grad(outputs=avg_a, inputs=input_data, grad_outputs=dummy, create_graph=True)[0]
    actual_da_dt = actual_da[:, 3]
    actual_db = torch.autograd.grad(outputs=avg_b, inputs=input_data, grad_outputs=dummy, create_graph=True)[0]
    actual_db_dt = actual_db[:, 3]
    # 获取模型中的参数
    k1, k2, k3, k4 = model.k1, model.k2, model.k3, model.k4
    
    dx_dt_theoretical = k1 * avg_x**2 * avg_a - k2 * avg_x**3 + k3 * avg_b - k4 * avg_x
    da_dt_theoretical = -k1 * avg_x**2 * avg_a + k2 * avg_x**3
    db_dt_theoretical = -k3 * avg_b + k4 * avg_x
    loss_x = torch.mean((actual_dx_dt - dx_dt_theoretical)**2)
    loss_a = torch.mean((actual_da_dt - da_dt_theoretical)**2)
    loss_b = torch.mean((actual_db_dt - db_dt_theoretical)**2)

    return loss_x + loss_a + loss_b



# Hyperparameters
alpha, beta, gamma = 0.0, 1, 1

num_epochs = 10000
  # Example values for k1, k2, k3, k4

model = PINN()
lambda_model = LambdaNetwork()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer_lambda = optim.Adam(lambda_model.parameters(), lr=1e-3)

training_data = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    optimizer_lambda.zero_grad()
    
    input_tensor.requires_grad_(True)
    p_pred = model(input_tensor)

    ones = torch.ones(p_pred.shape, requires_grad=False)
    #actual_dx_dt = torch.autograd.grad(outputs=p_pred, inputs=t_tensor, grad_outputs=ones, create_graph=True)[0]
    grad_outputs = torch.autograd.grad(outputs=p_pred, inputs=input_tensor, 
                                       grad_outputs=ones, create_graph=True)[0]
    #input(">>> ")
    lambda_values = lambda_model(t_tensor[:, None])
    p_pred = p_pred.squeeze(-1)
    # Calculate predicted 1st and 2nd order moments
    predicted_moment_x_1 = calculate_moments(p_pred, x_tensor, 1)
    predicted_moment_x_2 = calculate_moments(p_pred, x_tensor, 2)
    predicted_moment_a_1 = calculate_moments(p_pred, a_tensor, 1)
    predicted_moment_a_2 = calculate_moments(p_pred, a_tensor, 2)
    predicted_moment_b_1 = calculate_moments(p_pred, b_tensor, 1)
    predicted_moment_b_2 = calculate_moments(p_pred, b_tensor, 2)

    # Calculate actual 1st and 2nd order moments from the data
    actual_moment_x_1 = calculate_moment(x_tensor, 1)
    actual_moment_x_2 = calculate_moment(x_tensor, 2)
    actual_moment_a_1 = calculate_moment(a_tensor, 1)
    actual_moment_a_2 = calculate_moment(a_tensor, 2)
    actual_moment_b_1 = calculate_moment(b_tensor, 1)
    actual_moment_b_2 = calculate_moment(b_tensor, 2)

    # Apply lambda values to moment losses
    
    weighted_moment_loss_x_1 = lambda_values[:, 0] * moment_loss(predicted_moment_x_1, actual_moment_x_1)
    weighted_moment_loss_x_2 = lambda_values[:, 1] * moment_loss(predicted_moment_x_2, actual_moment_x_2)
    weighted_moment_loss_a_1 = lambda_values[:, 2] * moment_loss(predicted_moment_a_1, actual_moment_a_1)
    weighted_moment_loss_a_2 = lambda_values[:, 3] * moment_loss(predicted_moment_a_2, actual_moment_a_2)
    weighted_moment_loss_b_1 = lambda_values[:, 4] * moment_loss(predicted_moment_b_1, actual_moment_b_1)
    weighted_moment_loss_b_2 = lambda_values[:, 5] * moment_loss(predicted_moment_b_2, actual_moment_b_2)
    
    
    # Total moment loss
    total_moment_loss = (torch.sum(weighted_moment_loss_x_1) +
                         torch.sum(weighted_moment_loss_x_2) +
                         torch.sum(weighted_moment_loss_a_1) +
                         torch.sum(weighted_moment_loss_a_2) +
                         torch.sum(weighted_moment_loss_b_1) +
                         torch.sum(weighted_moment_loss_b_2))
    
    entropy_loss_val = max_entropy_loss(p_pred)
    # Calculate physics-based loss (requires implementation of physics_loss)
    #physics_loss_val = physics_loss(p_pred, x_tensor, a_tensor, b_tensor, t_tensor, model)
    physics_loss_val = physics_loss(p_pred, input_tensor, model)
    #  + gamma * physics_loss_val
    total_loss = -alpha * entropy_loss_val + beta * total_moment_loss + gamma * physics_loss_val
    total_loss.backward()
    optimizer.step()
    optimizer_lambda.step()
    training_data.append((epoch, total_loss.item(), model.k1.item(), model.k2.item(), model.k3.item(), model.k4.item()))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss.item()}, physics_loss_val: {physics_loss_val.item()}")
        print(f"k1: {model.k1.item()}, k2: {model.k2.item()}, k3: {model.k3.item()}, k4: {model.k4.item()}")

with open('training_data.txt', 'w') as file:
    file.write('Epoch, Loss, k1, k2, k3, k4\n')  # Header
    for data in training_data:
        file.write(', '.join(map(str, data)) + '\n')



import matplotlib.pyplot as plt

# Assuming final_p_pred contains the final predicted probabilities
# and losses contains the loss at each epoch
final_p_pred = model(input_tensor)
# Plotting the predicted probabilities P
plt.figure(figsize=(10, 6))
plt.plot(final_p_pred.detach().numpy(), label='Predicted Probabilities P')
plt.title('Predicted Probabilities P')
plt.xlabel('Data Point Index')
plt.ylabel('Probability')
plt.legend()
plt.show()

# Plotting the loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
