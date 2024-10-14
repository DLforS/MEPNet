# -*- coding: utf-8 -*-
# @Time    : 2024/4/23 23:15
# @Author  : Jay
# @File    : 3d_test.py
# @Project: gauss2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


def mixture_multivariate_normal_pdf(x, alpha1, mu1, cov1, alpha2, mu2, cov2):
    pdf1 = multivariate_normal.pdf(x, mean=mu1, cov=cov1)
    pdf2 = multivariate_normal.pdf(x, mean=mu2, cov=cov2)
    return alpha1 * pdf1 + alpha2 * pdf2


# 设置组分的参数
alpha1 = 0.6
mu1 = np.array([0.05, 0.05])
cov1 = np.array([[0.1, 0.01],
                 [0.01, 0.1]])

alpha2 = 0.4
mu2 = np.array([0.75, 0.75])
cov2 = np.array([[0.1, -0.05],
                [-0.05, 0.1]])

# 生成网格点
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# 计算概率密度函数值
Z = mixture_multivariate_normal_pdf(pos, alpha1, mu1, cov1, alpha2, mu2, cov2)

# 绘制等高线图
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mixture of Multivariate Normal Distribution')
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')
ax.set_title('Mixture of Multivariate Normal Distribution (3D Surface)')
plt.show()
