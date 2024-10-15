import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io

# Load the data
data_path = 'Y_samples_data.mat'
data = scipy.io.loadmat(data_path)
Y_samples = data['Y_samples']

# Extract the data
x_data = Y_samples[0, :, :]
a_data = Y_samples[1, :, :]
b_data = Y_samples[2, :, :]
num_time_points = Y_samples.shape[2]
time = np.linspace(0, 10, num_time_points)

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)

# Initialize the network
pinn = PINN()

# Define an optimizer
optimizer = optim.Adam(pinn.parameters(), lr=0.001)

# Preparing the data
x_flat = x_data.flatten()
a_flat = a_data.flatten()
b_flat = b_data.flatten()
t_flat = np.tile(time, (x_data.shape[0], 1)).flatten()
input_data = np.column_stack([x_flat, a_flat, b_flat, t_flat])
input_tensor = torch.tensor(input_data, dtype=torch.float32)

# Function to compute derivatives
def compute_derivatives(model, input_tensor):
    input_tensor.requires_grad_(True)
    p_pred = model(input_tensor)
    ones = torch.ones(p_pred.shape, requires_grad=False)
    grad_outputs = torch.autograd.grad(outputs=p_pred, inputs=input_tensor, 
                                       grad_outputs=ones, create_graph=True)[0]
    dp_dx = grad_outputs[:, 0]
    dp_da = grad_outputs[:, 1]
    dp_db = grad_outputs[:, 2]
    dp_dt = grad_outputs[:, 3]
    return dp_dx, dp_da, dp_db, dp_dt

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    p_pred = pinn(input_tensor)

    # Compute derivatives
    dp_dx, dp_da, dp_db, dp_dt = compute_derivatives(pinn, input_tensor)

    # Physical loss (assuming derivatives should be close to zero)
    physical_loss = torch.mean(dp_dx**2) + torch.mean(dp_da**2) + torch.mean(dp_db**2) + torch.mean(dp_dt**2)

    # Total loss (you might have additional data-driven loss terms)
    loss = physical_loss

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Print loss
    if epoch % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
