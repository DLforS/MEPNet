% 导入数据文件
data0 = importdata('key_moment_0_data.txt');
data25 = importdata('key_moment_25_data.txt');
data49 = importdata('key_moment_49_data.txt');

% 提取数据
x_values0 = load('xvalues.txt');
Z0 = data0(1:148,:);
predictions_df0 = data0(149:end,:);

x_values25 = load('xvalues.txt');
Z25 = data25(1:148,:);
predictions_df25 = data25(149:end,:);

x_values49 = load('xvalues.txt');
Z49 = data49(1:148,:);
predictions_df49 = data49(149:end,:);

% 创建包含三个子图的图形窗口
figure;

% 绘制第一个时刻的图像
subplot(1, 3, 1);
plot(x_values0, Z0, 'b-', 'LineWidth', 1.5);
hold on;
plot(x_values0, predictions_df0, 'r--', 'LineWidth', 1.5);
legend('True Values', 'Predicted Values', 'Location', 'best');
title('Key Moment 0');
xlabel('X Values');
ylabel('Values');
grid on;

% 绘制第二个时刻的图像
subplot(1, 3, 2);
plot(x_values25, Z25, 'b-', 'LineWidth', 1.5);
hold on;
plot(x_values25, predictions_df25, 'r--', 'LineWidth', 1.5);
legend('True Values', 'Predicted Values', 'Location', 'best');
title('Key Moment 25');
xlabel('X Values');
ylabel('Values');
grid on;

% 绘制第三个时刻的图像
subplot(1, 3, 3);
plot(x_values49, Z49, 'b-', 'LineWidth', 1.5);
hold on;
plot(x_values49, predictions_df49, 'r--', 'LineWidth', 1.5);
legend('True Values', 'Predicted Values', 'Location', 'best');
title('Key Moment 49');
xlabel('X Values');
ylabel('Values');
grid on;
