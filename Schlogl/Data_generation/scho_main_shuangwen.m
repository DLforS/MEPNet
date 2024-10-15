clear all, clc
% 初始化条件
x0 = [250, 1e5, 2e5];  % [X, A, B] 初始条件
tspan = [0:0.1:50];

% ODE求解
[t, x] = ode23s(@(t, x) scho(t, x), tspan, x0);

% 绘图
figure
hold on
plot(t, x(:,1)) % 只绘制X的变化，因为A和B是常数
legend('X')
xlabel('Time')
ylabel('Concentration')
title('Schlögl Model Simulation')

% ODE函数
function dy = scho(t, y)
    % 固定A和B的值
    A = 1e5; % A的常数值
    B = 2e5; % B的常数值

    % 反应速率常数
    k1 = 3e-7/2; k2 = 1e-4/6; k3 = 1e-3; k4 = 3.5;

    % 反应速率方程
    dy = zeros(3,1);
    dy(1) = k1 * y(1)^2 * A - k2 * y(1)^3 + k3 * B - k4 * y(1); % X的变化率
    dy(2) = 0; % A保持不变
    dy(3) = 0; % B保持不变
end
