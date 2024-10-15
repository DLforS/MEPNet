% Simulate a two-state model of cell differentiation
clear all;clc;
import Gillespie.*

%  Schl?gl model:
%   1. 2X + A --k1---------> 3X
%   2. 3X --k2--> 2X + A
%   3. B --k3---------> X
%   4. X --k4------------> B
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p.k1 = 0.15;      
% p.k2 = 0.0015;
% p.k3 = 20;
% p.k4 = 3.5;
p.k1 = 3e-7/2;      
p.k2 = 1e-4/6;
p.k3 = 1e-3;
p.k4 = 3.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial state
% tspan = [0,50]; %seconds
% tspan = linspace(0, 5, 10);
x0 = [250,1e5,2e5];     % X A B
% Specify reaction network
pfun = @propensities_2state;
stoich_matrix = [ 1  0  0   
                  -1  0  0
                  1   0  0 
                  -1  0  0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number_of_Gillespie_runs = 10;
% for i=1:1:number_of_Gillespie_runs
%     [t,x] = firstReactionMethod(stoich_matrix, pfun, tspan, x0, p);
%     for j = 1:1:length(x)
%         for k=1:1:2
%             list(i,j,k)=x(j,k);
%             list2(i,j)=t(j);
%         end
%     end
% end
% %%%%%%%%determine%%%%%%%
% for i=2:1:length(list2(10,:))
%     if list2(10,i)==0
%         i-1
%         break
%     end
% end
% % %%%%%%%%%%%%%%%%%%%%%%
% figure
% hold on
% stairs(list2(1,:),list(1,:,1))
% stairs(list2(2,:),list(2,:,1))
% stairs(list2(3,:),list(3,:,1))
% stairs(list2(4,:),list(4,:,1))
% stairs(list2(5,:),list(5,:,1))
% stairs(list2(6,:),list(6,:,1))
% stairs(list2(7,:),list(7,:,1))
% stairs(list2(8,:),list(8,:,1))
% stairs(list2(9,:),list(9,:,1))
% stairs(list2(10,:),list(10,:,1))
% figure
% hold on
% stairs(list2(1,:),list(1,:,1))
% stairs(list2(2,:),list(2,:,1))
% stairs(list2(3,:),list(3,:,1))
% stairs(list2(4,:),list(4,:,1))
% stairs(list2(5,:),list(5,:,1))
% stairs(list2(6,:),list(6,:,1))
% stairs(list2(7,:),list(7,:,1))
% stairs(list2(8,:),list(8,:,1))
% stairs(list2(9,:),list(9,:,1))
% stairs(list2(10,:),list(10,:,1))
% Plot time courses
% figure();
% [t,x] = firstReactionMethod(stoich_matrix, pfun, tspan, x0, p);
% hold on
% stairs(t,x(:,1)); 
% stairs(t,x(:,2)); 
% stairs(t,x(:,3)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 1e6;
aa = zeros(n,4);
count = 0;
% for j=0:50:50
%  for j=0.01:10:100
% 时间点设置
% sample_times = [0.01,0.05,0.1,0.5,1,2,3,4,5]; % 从 0 到 50 的时间点
sample_times = 5; % 从 0 到 50 的时间点
num_time_points = length(sample_times);
min_values = zeros(num_time_points, 1);  % 初始化存储最小值的数组
max_values = zeros(num_time_points, 1);  % 初始化存储最大值的数组
num_runs = 1e4; % 每个时间点的模拟次数
Y_samples = zeros(num_time_points, 3, num_runs); % 初始化结果数组

% 在每个时间点进行 SSA 模拟
for i = 1:num_time_points
    current_time = sample_times(i);
    i
    for j = 1:num_runs
        [t, YY] = firstReactionMethod(stoich_matrix, pfun, [0, current_time], x0, p);
        YY = YY';
        Y_samples(i, :, j) = YY(:, end); % 保存每个时间点的模拟结果
    end
end
N=100;
all_histogram_values = zeros(num_time_points, N); % 使用零初始化
for i = 1:num_time_points
    data = squeeze(Y_samples(i, 1, :));
    min_values(i) = min(data);  % 计算并存储当前数据的最小值
    max_values(i) = max(data);  % 计算并存储当前数据的最大值
%     figure;
    %histogram(Y_samples(i, 1, :));
    A = histogram(data, [min(data):5:max(data)], 'Normalization', 'probability');
%     title(['Distribution at time ', num2str(sample_times(i))]);
    histValues = A.Values;
    all_histogram_values(i, 1:length(histValues)) = histValues; % 仅存储直方图的值
end
% 保存所有时间点的直方图值到文本文件
save('all_histogram_values_1e4.txt', '-ascii', 'all_histogram_values');
min_max_values = [min_values, max_values];
save('min_max_values_1e4.txt', '-ascii', 'min_max_values');
% % 如果需要，可以保存 Y_samples 数组
% % save('Y_samples.mat', 'Y_samples');
% 
% % 假设 sample_times 是时间点数组，Y_samples 是模拟数据
% num_time_points = length(sample_times);
% num_runs = size(Y_samples, 3); % 模拟运行次数
% 
% % 初始化存储平均值的数组
% mean_values = zeros(num_time_points, 1);
% 
% % 计算每个时间点的平均值
% for i = 1:num_time_points
%     mean_values(i) = mean(Y_samples(i, 1, :));
% end
% 
% % 绘制趋势图
% figure;
% hold on;
% 
% % 绘制每次模拟的结果
% for i = 1:num_time_points
%     for j = 1:num_runs
%         plot(sample_times(i), Y_samples(i, 1, j), 'b.'); % 'b.' 表示蓝色点
%     end
% end
% 
% % 绘制平均值线
% plot(sample_times, mean_values, 'r-o', 'LineWidth', 2); % 'r-o' 表示红色圆圈标记和线
% 
% xlabel('Time');
% ylabel('Concentration of Species 1');
% title('Trend of Species 1 Over Time with Individual Simulation Data');
% legend('Individual Runs', 'Mean Value');
% 
% 
