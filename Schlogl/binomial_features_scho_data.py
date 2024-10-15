import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
from scipy.special import comb

# 读取最小值和最大值
min_max_values = np.loadtxt('min_max_values_1e4.txt')
min_values = min_max_values[:, 0]
max_values = min_max_values[:, 1]

# 定义归一化和反归一化函数
def normalize_data(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)

def denormalize_data(norm_data, min_value, max_value):
    return norm_data * (max_value - min_value) + min_value

# 生成二项式特征，并包含时间维度
def binomial_features_time(t_values, degree=10):
    all_features = []
    for idx, t in enumerate(t_values):
        x_values = np.arange(min_values[idx], max_values[idx] + 5, 5)  # 动态生成 x_values
        x_norm = normalize_data(x_values, min_values[idx], max_values[idx])  # 归一化
        features_t = [comb(degree, k) * (x_norm ** k) * ((1 - x_norm) ** (degree - k)) for k in range(degree + 1)]
        features_t = np.array(features_t).T
        time_c = np.full((features_t.shape[0], 1), t)
        features_time = np.hstack([features_t, time_c])
        all_features.append(features_time)
    features = np.concatenate(all_features, axis=1)
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

def loss_func(model, moments, x, true_px):
    px = model(x).squeeze()
    entropy = torch.sum(px * torch.log(px + 1e-10))
    moment_constraints = torch.sum(px[:, :, None] * x[:, :, 0:-1], dim=0) - moments
    mse = torch.mean((px - true_px.squeeze())**2)
    loss = torch.mean(moment_constraints**2) - entropy
    return loss, mse, entropy, px

# 读取直方图数据
with open('all_histogram_values_1e4.txt', 'r') as file:
    file_content = file.readlines()
histogram = [np.array(line.split(), dtype=float) for line in file_content]
histogram = np.array(histogram).T

t_values = np.linspace(0.1, 5, 50)  # 时间点

# 生成归一化特征
input_tensor = binomial_features_time(t_values)
gauss_tensor = torch.tensor(histogram, dtype=torch.float32).unsqueeze(-1)  # 添加维度以适配模型

# 计算矩
moments = []
for i in range(input_tensor.shape[2]-1):
    moment = torch.sum(gauss_tensor * input_tensor[:, :, i:i+1], dim=0)
    moments.append(moment)
moments = torch.cat(moments, dim=1)

# 初始化模型和优化器
time_dimension = t_values.shape[0]
feature_dimension = input_tensor.shape[1]  # 由于 x_values 的长度可能不同，特征维度直接从 input_tensor 获取
input_dim = feature_dimension * time_dimension

model = MEPNet(input_dim, time_dimension)
optimizer = Adam(model.parameters(), lr=0.001)

# 训练模型
loss_history = []
mse_history = []
entropy_history = []
min_mse = float('inf')
for step in range(100000):
    optimizer.zero_grad()
    loss, mse, entropy, px = loss_func(model, moments, input_tensor, gauss_tensor)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    mse_history.append(mse.item())
    entropy_history.append(-entropy.item())

    if mse < min_mse:
        min_mse = mse
        best_model_params = model.state_dict().copy()

    if (step + 1) % 1000 == 0:
        print(f'Step {step+1}, Loss: {loss.item()}, MSE: {mse.item()}, Entropy: {-entropy.item()}')

# 模型预测和结果反归一化
def model_predictions(model, t_values):
    predictions = []
    for idx, t in enumerate(t_values):
        x_values = np.arange(min_values[idx], max_values[idx] + 5, 5)
        input_tensor = binomial_features_time([t])  # 生成当前时间点的特征
        model.eval()
        with torch.no_grad():
            pred = model(input_tensor)
        pred = pred.numpy().flatten()
        denorm_pred = denormalize_data(pred, min_values[idx], max_values[idx])
        predictions.append(denorm_pred)
    return np.array(predictions)

model.load_state_dict(best_model_params)
model_pred = model_predictions(model, t_values)

# 绘制结果
plt.figure(figsize=(15, 5))
for i, t in enumerate(t_values):
    plt.subplot(5, 10, i+1)
    plt.plot(np.arange(min_values[i], max_values[i] + 5, 5), model_pred[i], 'r-', label='Prediction')
    plt.plot(np.arange(min_values[i], max_values[i] + 5, 5), histogram[:, i], 'b-', label='True')
    plt.title(f'Time {t:.1f}')
    plt.legend()
plt.tight_layout()
plt.show()
