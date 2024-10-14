# -*- coding: utf-8 -*-
# @Time    : 2024/4/27 15:02
# @Author  : Jay
# @File    : 3d_plot.py
# @Project: High_dimensional_case
# 将结果进行可视化
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    path = r'C:\Users\Rubis\Desktop\Documents\MaxEntNet\Example\High_dimensional_case\three_dimen'
    file_name = r'\3d_gauss_pred.npz'
    data = np.load(path+file_name)

    X, Y, Z, pdf, pred_pdf = data['arr1'], data['arr2'], data['arr3'], data['arr4'], data['arr5']
    # 生成网格点
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    # z = np.linspace(0, 1, 101)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Plotting contourf
    z_index = 50  # 固定的 Z 索引值
    contour = axs[0].contourf(X[:, :, z_index], Y[:, :, z_index], pred_pdf[:, :, z_index], cmap='viridis')
    cb = fig.colorbar(contour, ax=axs[0])  # 添加颜色条
    cb.ax.tick_params(labelsize=18)
    axs[0].set_xlabel('X', fontsize=25)
    axs[0].set_ylabel('Y', fontsize=25)
    axs[0].set_title('MEP-Net', fontsize=25)
    axs[0].tick_params(labelsize=18)

    x_index = 50  # 固定的 X 索引值
    contour = axs[1].contourf(X[:, :, z_index], Y[:, :, z_index], pred_pdf[x_index, :, :], cmap='viridis')
    cb = fig.colorbar(contour, ax=axs[1])  # 添加颜色条
    cb.ax.tick_params(labelsize=18)
    axs[1].set_xlabel('Y', fontsize=25)
    axs[1].set_ylabel('Z', fontsize=25)
    # axs[1].set_title('MEP-Net}', fontsize=25)
    axs[1].tick_params(labelsize=18)

    y_index = 50  # 固定的 Y 索引值
    contour = axs[2].contourf(X[:, :, z_index], Y[:, :, z_index], pred_pdf[:, y_index, :], cmap='viridis')
    ch = fig.colorbar(contour, ax=axs[2])  # 添加颜色条
    ch.ax.tick_params(labelsize=18)
    axs[2].set_xlabel('X', fontsize=25)
    axs[2].set_ylabel('Z', fontsize=25)
    # axs[2].set_title('MEP-Net', fontsize=25)
    axs[2].tick_params(labelsize=18)

    plt.tight_layout()
    plt.show()
    fig.savefig('figure_2.png', dpi=600)

    # Plot_error
    fig1, axs1 = plt.subplots(1, 3, figsize=(20, 6))

    z_index = 50  # 固定的 Z 索引值
    contour = axs1[0].contourf(X[:, :, z_index], Y[:, :, z_index], np.abs(pred_pdf[:, :, z_index] - pdf[:, :, z_index]),
                               cmap='viridis')
    cb = fig1.colorbar(contour, ax=axs1[0])  # 添加颜色条
    cb.ax.tick_params(labelsize=18)
    axs1[0].set_xlabel('X', fontsize=25)
    axs1[0].set_ylabel('Y', fontsize=25)
    axs1[0].set_title('Error', fontsize=25)
    axs1[0].tick_params(labelsize=18)

    x_index = 50  # 固定的 X 索引值
    contour = axs1[1].contourf(X[:, :, z_index], Y[:, :, z_index], np.abs(pred_pdf[x_index, :, :] - pdf[x_index, :, :]),
                               cmap='viridis')
    cb = fig1.colorbar(contour, ax=axs1[1])  # 添加颜色条
    cb.ax.tick_params(labelsize=18)
    axs1[1].set_xlabel('Y', fontsize=25)
    axs1[1].set_ylabel('Z', fontsize=25)
    # axs1[1].set_title('Error', fontsize=25)
    axs1[1].tick_params(labelsize=18)

    y_index = 50  # 固定的 Y 索引值
    contour = axs1[2].contourf(X[:, :, z_index], Y[:, :, z_index], np.abs(pred_pdf[:, y_index, :] - pdf[:, y_index, :]),
                               cmap='viridis')
    ch = fig1.colorbar(contour, ax=axs1[2])  # 添加颜色条
    ch.ax.tick_params(labelsize=18)
    axs1[2].set_xlabel('X', fontsize=25)
    axs1[2].set_ylabel('Z', fontsize=25)
    # axs1[2].set_title('Error', fontsize=25)
    axs1[2].tick_params(labelsize=18)

    plt.tight_layout()
    plt.show()
    fig1.savefig('figure_3.png', dpi=600)


