# -*- coding: utf-8 -*-
# @Time    : 2024/4/24 15:29
# @Author  : Jay
# @File    : MEPNet_2d.py
# @Project: High_dimensional_case
# 尝试利用Max Entropy Neural Network 拟合二维混合高斯分布的概率密度函数
import time
import torch
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import comb
from src.MEP_Net import MaxEntNet
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 生成网格点
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    x_tensor = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

    # 计算概率密度函数值
    Z = mul_gauss(pos)
    gauss_tensor = np.array(Z.reshape(-1, 1))

    # binomial_features
    n = 10
    binomial_features = [comb(n, k) * comb(n, v) * (x_tensor[:, 0:1] ** k) * ((1 - x_tensor[:, 0:1]) ** (n - k)) * (
                x_tensor[:, 1:2] ** v) * ((1 - x_tensor[:, 1:2]) ** (n - v)) for k in range(n + 1) for v in
                         range(n + 1)]
    x_tensor = np.concatenate(binomial_features, axis=-1).squeeze()

    # initial_ent_weight, max_iterations, min_mse, layers, lr, optimizer_name
    initial_ent_weight = 1e-6
    max_iterations = 10000
    min_mse = float('inf')
    layers = [x_tensor.shape[-1], 64, 64, 64, 64, 1]
    lr = 0.001
    optimizer_name = 'Adam'

    # Train and Predict
    model = MaxEntNet(x_tensor, gauss_tensor, initial_ent_weight, max_iterations, min_mse, layers, lr, optimizer_name,
                      device)
    start_time = time.time()
    ent_history, loss_history, mse_history = model.train()
    end_time = time.time()
    print("总计耗时: {:.2f}".format(end_time - start_time))
    px = model.predict(x_tensor)
    pred_Z = px.reshape(Z.shape[0], Z.shape[1])

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex='col', sharey='row')

    # Plotting contourf
    contour = axs[0].contourf(X, Y, Z, cmap='viridis')
    cb = fig.colorbar(contour, ax=axs[0])  # 添加颜色条
    cb.ax.tick_params(labelsize=18)
    axs[0].set_xlabel('X', fontsize=25)
    axs[0].set_ylabel('Y', fontsize=25)
    axs[0].set_title('Mixed Gaussian', fontsize=25)
    axs[0].tick_params(labelsize=18)

    contour = axs[1].contourf(X, Y, pred_Z, cmap='viridis')
    cb = fig.colorbar(contour, ax=axs[1])  # 添加颜色条
    cb.ax.tick_params(labelsize=18)
    # axs[1].set_xlabel('X', fontsize=25)
    # axs[1].set_ylabel('Y', fontsize=25)
    axs[1].set_title('MEP-Net', fontsize=25)
    axs[1].tick_params(labelsize=18)

    contour = axs[2].contourf(X, Y, np.abs(pred_Z - Z), cmap='viridis')
    ch = fig.colorbar(contour, ax=axs[2])  # 添加颜色条
    ch.ax.tick_params(labelsize=18)
    # axs[2].set_xlabel('X', fontsize=25)
    # axs[2].set_ylabel('Y', fontsize=25)
    axs[2].set_title('Error', fontsize=25)
    axs[2].tick_params(labelsize=18)

    plt.tight_layout()
    plt.show()
    fig.savefig('figure_1.png', dpi=300)
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(35, 10))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 20      
    # Plotting 3D surface for the true PDF
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_xlabel('X', fontsize=30, labelpad=30)
    ax1.set_ylabel('Y', fontsize=30, labelpad=30)
    ax1.set_zlabel('Density', fontsize=30, labelpad=30)
    ax1.set_title('2-dim Gaussian', fontsize=50)
    ax1.tick_params(labelsize=30)
    ax1.legend()
    #fig.colorbar(surf1, shrink=0.5)

    # Plotting 3D surface for the predicted PDF
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, pred_Z, cmap='viridis')
    ax2.set_xlabel('X', fontsize=30, labelpad=30)
    ax2.set_ylabel('Y', fontsize=30, labelpad=30)
    #ax2.set_zlabel('Density', fontsize=30, labelpad=10)
    ax2.set_title('MEP-Net', fontsize=50)
    ax2.tick_params(labelsize=30)
    ax2.legend()
    
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, np.abs(pred_Z-Z), cmap='viridis')
    ax3.set_xlabel('X', fontsize=30, labelpad=30)
    ax3.set_ylabel('Y', fontsize=30, labelpad=30)
    #ax3.set_zlabel('Density', fontsize=30, labelpad=30)
    ax3.set_title('Error', fontsize=50)
    ax3.tick_params(labelsize=30)
    ax3.legend()
    
    #fig.colorbar(surf2, shrink=0.5)
    # Adjust the position of the colorbar
    plt.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(surf2, shrink=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(surf2, shrink=0.8)
    cbar.ax.tick_params(labelsize=30)
    plt.tight_layout()
    plt.show()
    fig.savefig('Gaussian_2dim.png', dpi=300)


    plt.figure()
    plt.plot(ent_history, 'm-')
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(np.log(loss_history), 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.close()
