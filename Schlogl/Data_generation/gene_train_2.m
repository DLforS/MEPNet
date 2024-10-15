% Simulate a two-state model of cell differentiation
clear all;clc;
import Gillespie.*

%  Schl?gl model:
%   1. 2X + A --k1---------> 3X
%   2. 3X --k2--> 2X + A
%   3. B --k3---------> X
%   4. X --k4------------> B
%%%%%%%%%%%%%%%%%%%%%%%%%%%
p.k1 = 0.0015;      
p.k2 = 0.15;
p.k3 = 20;
p.k4 = 3.5;
% k1 = 0.0015;      
% k2 = 0.15;
% k3 = 20;
% k4 = 3.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial state
% tspan = [0,50]; %seconds
% tspan = linspace(0, 5, 10);
x0 = [25,0,0];     % X A B
% Specify reaction network
pfun = @propensities_2state;
stoich_matrix = [ 1  -1   0   
                  -1  1   0
                  1   0  -1 
                  -1  0   1]; 
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
n = 1000;
aa = zeros(n, 6); % 存储均值、一阶矩到五阶矩，共6个统计量
number_of_Gillespie_runs = n;
number_of_time_points = 100;
Y_samples = zeros(3, number_of_Gillespie_runs, number_of_time_points); % 假设3个状态
time_points = linspace(0.01, 10, number_of_time_points);

for i = 1:number_of_Gillespie_runs
    sample_time = [0, time_points(end)]; % 只考虑最后一个时间点
    [t, YY] = firstReactionMethod(stoich_matrix, pfun, sample_time, x0, p);
    YY = YY';
    Y_samples(:, i, end) = YY(:, end); % 存储最后一个时间点的结果
end

% 计算每个状态的均值和矩
% for state = 1:size(Y_samples, 1)
    final_samples = squeeze(Y_samples(2, :, end)); % 提取特定状态的最后一个时间点的所有样本
    aa(:, 1) = mean(final_samples);    
    aa(:, 2) = mean(final_samples.^2);
    aa(:, 3) = mean(final_samples.^3);
    aa(:, 4) = mean(final_samples.^4);
    aa(:, 5) = mean(final_samples.^5);
% end




