import time
import torch
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import comb
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn as nn
from torch.optim import Adam

def mul_gauss(x):
    # Setting basic parameters
    alpha1 = 0.6
    mu1 = np.array([0.05, 0.05])
    cov1 = np.array([[0.1, 0.01],
                     [0.01, 0.1]])

    alpha2 = 0.4
    mu2 = np.array([0.75, 0.75])
    cov2 = np.array([[0.1, -0.05],
                     [-0.05, 0.1]])

    pdf1 = multivariate_normal.pdf(x, mean=mu1, cov=cov1)
    pdf2 = multivariate_normal.pdf(x, mean=mu2, cov=cov2)

    return alpha1 * pdf1 + alpha2 * pdf2

class DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(
                ('activation_%d' % i, self.activation())
            )

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1], bias=False))
        )
        layer_list.append(
            ('activation_%d' % (self.depth - 1), torch.nn.Softplus())
        )
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

class MaxEntNetWithEntropy:
    def __init__(self, x, true_px, initial_ent_weight, max_iterations, min_mse, layers, lr, device):
        self.device = device
        self.x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).clone().detach()
        self.gauss_tensor = torch.as_tensor(true_px, dtype=torch.float32, device=self.device).clone().detach()
        self.min_mse = min_mse

        self.net = DNN(layers)
        self.net.to(self.device)

        self.moments = self.cal_moments(self.gauss_tensor, self.x_tensor)

        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.initial_ent_weight = initial_ent_weight
        self.max_iterations = max_iterations

        self.ent_history = []
        self.loss_history = []
        self.mse_history = []
        self.best_model_params = None

        print("Initialization Completed!")

    def cal_moments(self, gauss_tensor, x_tensor):
        moments = list()
        for i in range(x_tensor.shape[-1]):
            moment = torch.sum(gauss_tensor * x_tensor[:, i:i + 1], dim=0)
            moments.append(moment)
        moments = torch.cat(moments, dim=0)
        return moments

    def adjust_ent_weight(self, initial_ent_weight, current_iterations, max_iterations):
        alpha = 0.1
        return initial_ent_weight * (alpha ** (current_iterations / max_iterations))

    def cal_entropy(self, px, prev_px):
        epsilon = 1e-10
        entropy = torch.sum(px * torch.log(px / (prev_px + epsilon)))
        return entropy

    def train(self):
        self.net.train()
        print("Training with Entropy Constraint!")
        prev_px = torch.ones_like(self.gauss_tensor.squeeze())
        for step in range(self.max_iterations):
            ent_weight = self.adjust_ent_weight(self.initial_ent_weight, step, self.max_iterations)
            self.optimizer.zero_grad()
            px = self.net(self.x_tensor).squeeze()
            ent = self.cal_entropy(px, prev_px)
            moment_constraints = torch.sum(px[:, None] * self.x_tensor, dim=0) - self.moments
            loss = torch.mean(moment_constraints ** 2) + 1e-6 * ent
            mse = torch.mean((px - self.gauss_tensor.squeeze()) ** 2)

            if loss.requires_grad:
                loss.backward()
            self.optimizer.step()

            prev_px = px.detach()

            if mse.item() < self.min_mse:
                self.min_mse = mse.item()
                self.best_model_params = self.net.state_dict().copy()

            if (step + 1) % 1000 == 0:
                print(f'Step {step + 1}, Loss: {loss.item()}, Entropy: {ent.item()}, MSE: {mse.item()}')

            self.ent_history.append(ent.item())
            self.loss_history.append(loss.item())
            self.mse_history.append(mse.item())
        return self.ent_history, self.loss_history, self.mse_history

    def predict(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device).clone().detach()

        self.net.load_state_dict(self.best_model_params)
        self.net.eval()
        px = self.net(x)
        px = px.detach().cpu().numpy()

        return px

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Generate grid points
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    x_tensor = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

    # Calculate probability density function values
    Z = mul_gauss(pos)
    gauss_tensor = np.array(Z.reshape(-1, 1))

    # Binomial features
    n = 10
    binomial_features = [comb(n, k) * comb(n, v) * (x_tensor[:, 0:1] ** k) * ((1 - x_tensor[:, 0:1]) ** (n - k)) * (
                x_tensor[:, 1:2] ** v) * ((1 - x_tensor[:, 1:2]) ** (n - v)) for k in range(n + 1) for v in
                         range(n + 1)]
    x_tensor = np.concatenate(binomial_features, axis=-1).squeeze()

    # Set hyperparameters
    initial_ent_weight = 1e-6
    max_iterations = 10000
    min_mse = float('inf')
    layers = [x_tensor.shape[-1], 64, 64, 64, 64, 1]
    lr = 0.001

    # Train and Predict
    model = MaxEntNetWithEntropy(x_tensor, gauss_tensor, initial_ent_weight, max_iterations, min_mse, layers, lr, device)
    start_time = time.time()
    ent_history, loss_history, mse_history = model.train()
    end_time = time.time()
    print("Total time: {:.2f} seconds".format(end_time - start_time))
    px = model.predict(x_tensor)
    pred_Z = px.reshape(Z.shape[0], Z.shape[1])

    # Save training records
    with open("training_records_2d_noentropy.txt", "w") as file:
        file.write("Step\tLoss\tMSE\tEntropy\n")
        for step in range(len(loss_history)):
            file.write(f"{step+1}\t{loss_history[step]}\t{mse_history[step]}\t{ent_history[step]}\n")

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex='col', sharey='row')

    # Plotting contourf
    contour = axs[0].contourf(X, Y, Z, cmap='viridis')
    cb = fig.colorbar(contour, ax=axs[0])
    cb.ax.tick_params(labelsize=18)
    axs[0].set_xlabel('X', fontsize=25)
    axs[0].set_ylabel('Y', fontsize=25)
    axs[0].set_title('Mixed Gaussian', fontsize=25)
    axs[0].tick_params(labelsize=18)

    contour = axs[1].contourf(X, Y, pred_Z, cmap='viridis')
    cb = fig.colorbar(contour, ax=axs[1])
    cb.ax.tick_params(labelsize=18)
    axs[1].set_xlabel('X', fontsize=25)
    axs[1].set_title('MEP-Net', fontsize=25)
    axs[1].tick_params(labelsize=18)

    contour = axs[2].contourf(X, Y, np.abs(pred_Z - Z), cmap='viridis')
    ch = fig.colorbar(contour, ax=axs[2])
    ch.ax.tick_params(labelsize=18)
    axs[2].set_xlabel('X', fontsize=25)
    axs[2].set_title('Error', fontsize=25)
    axs[2].tick_params(labelsize=18)

    plt.tight_layout()
    plt.show()
    fig.savefig('figure_1_entropy.png', dpi=300)

    # 3D plotting
    fig = plt.figure(figsize=(35, 10))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 20      

    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_xlabel('X', fontsize=30, labelpad=30)
    ax1.set_ylabel('Y', fontsize=30, labelpad=30)
    ax1.set_zlabel('Density', fontsize=30, labelpad=30)
    ax1.set_title('2-dim Gaussian', fontsize=50)
    ax1.tick_params(labelsize=30)

    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, pred_Z, cmap='viridis')
    ax2.set_xlabel('X', fontsize=30, labelpad=30)
    ax2.set_ylabel('Y', fontsize=30, labelpad=30)
    ax2.set_title('MEP-Net', fontsize=50)
    ax2.tick_params(labelsize=30)
    
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, np.abs(pred_Z-Z), cmap='viridis')
    ax3.set_xlabel('X', fontsize=30, labelpad=30)
    ax3.set_ylabel('Y', fontsize=30, labelpad=30)
    ax3.set_title('Error', fontsize=50)
    ax3.tick_params(labelsize=30)
    
    cbar = fig.colorbar(surf2, shrink=0.8)
    cbar.ax.tick_params(labelsize=30)
    plt.tight_layout()
    plt.show()
    fig.savefig('Gaussian_2dim_entropy.png', dpi=300)
