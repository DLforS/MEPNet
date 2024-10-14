# -*- coding: utf-8 -*-
# @Time    : 2024/4/26 23:19
# @Author  : Jay
# @File    : 3d_test.py
# @Project: High_dimensional_case
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def mixture_multivariate_normal_pdf_3d(x, alpha1, mu1, cov1, alpha2, mu2, cov2, alpha3, mu3, cov3):
    pdf1 = multivariate_normal.pdf(x, mean=mu1, cov=cov1)
    pdf2 = multivariate_normal.pdf(x, mean=mu2, cov=cov2)
    pdf3 = multivariate_normal.pdf(x, mean=mu3, cov=cov3)
    return alpha1 * pdf1 + alpha2 * pdf2 + alpha3 * pdf3


# 设置组分的参数
alpha1 = 0.4
mu1 = np.array([0.05, 0.05, 0.05])
cov1 = np.array([[0.1, 0, 0],
                 [0, 0.1, 0],
                 [0, 0, 0.1]])

alpha2 = 0.3
mu2 = np.array([0.2, 0.8, 0.2])
cov2 = np.array([[0.15, 0, 0],
                 [0, 0.05, 0],
                 [0, 0, 0.15]])

alpha3 = 0.3
mu3 = np.array([0.8, 0.2, 0.8])
cov3 = np.array([[0.05, 0, 0],
                 [0, 0.15, 0],
                 [0, 0, 0.1]])


# 生成网格点
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
z = np.linspace(0, 1, 101)
X, Y, Z = np.meshgrid(x, y, z)
pos = np.empty(X.shape + (3,))
pos[:, :, :, 0] = X
pos[:, :, :, 1] = Y
pos[:, :, :, 2] = Z

# 计算概率密度函数值
pdf = mixture_multivariate_normal_pdf_3d(pos, alpha1, mu1, cov1, alpha2, mu2, cov2, alpha3, mu3, cov3)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# Plotting contourf
z_index = 50  # 固定的 Z 索引值
contour = axs[0].contourf(X[:, :, z_index], Y[:, :, z_index], pdf[:, :, z_index], cmap='viridis')
cb = fig.colorbar(contour, ax=axs[0])  # 添加颜色条
cb.ax.tick_params(labelsize=18)
axs[0].set_xlabel('X', fontsize=25)
axs[0].set_ylabel('Y', fontsize=25)
axs[0].set_title('Mixed Gaussian', fontsize=25)
axs[0].tick_params(labelsize=18)

x_index = 50  # 固定的 X 索引值
contour = axs[1].contourf(X[:, :, z_index], Y[:, :, z_index], pdf[x_index, :, :], cmap='viridis')
cb = fig.colorbar(contour, ax=axs[1])  # 添加颜色条
cb.ax.tick_params(labelsize=18)
axs[1].set_xlabel('Y', fontsize=25)
axs[1].set_ylabel('Z', fontsize=25)
# axs[1].set_title('Mixed Gaussian', fontsize=25)
axs[1].tick_params(labelsize=18)

y_index = 50  # 固定的 Y 索引值
contour = axs[2].contourf(X[:, :, z_index], Y[:, :, z_index], pdf[:, y_index, :], cmap='viridis')
ch = fig.colorbar(contour, ax=axs[2])  # 添加颜色条
ch.ax.tick_params(labelsize=18)
axs[2].set_xlabel('X', fontsize=25)
axs[2].set_ylabel('Z', fontsize=25)
# axs[2].set_title('Mixed Gaussian', fontsize=25)
axs[2].tick_params(labelsize=18)

plt.tight_layout()
plt.show()
fig.savefig('figure_1.png', dpi=600)
