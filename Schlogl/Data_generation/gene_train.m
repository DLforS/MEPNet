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
aa = zeros(n,6);count = 0;
number_of_Gillespie_runs = n;
number_of_time_points = 100;  % Set the number of time points you want to sample
Y_samples = zeros(3, number_of_Gillespie_runs, number_of_time_points);
time_points = linspace(0.01, 10, number_of_time_points);  % Adjust as per your requirements

count = 0;
for j = time_points
    count = count + 1
for i = 1:number_of_Gillespie_runs
        sample_time = [0, j];
        [t, YY] = firstReactionMethod(stoich_matrix, pfun, sample_time, x0, p);
        YY = YY';
        [m, k] = size(YY);
        Y_samples(:, i, count) = YY(:, k);  % Storing the results at the current time point
end
end

filename = 'Y_samples_data.mat';  
save(filename, 'Y_samples'); 
% load('Y_samples_data.mat'); 

P_matrix = zeros(number_of_Gillespie_runs, number_of_time_points);
for t = 1:number_of_time_points
    for x = 1:number_of_Gillespie_runs
        state_count = sum(Y_samples(1, :, t) == x);
        P_matrix(x, t) = state_count / number_of_Gillespie_runs;
    end
end
time_vector = linspace(0.01, 10, number_of_time_points);  
run_index_vector = 1:number_of_Gillespie_runs;
figure;
contour(run_index_vector, time_vector, P_matrix);
xlabel('Gillespie Run Index');
ylabel('Time (t)');
title('Contour Plot of Probability P(x,t)');
colorbar;  

figure
subplot(1,2,1)
hold on
plot(Y_samples); 
subplot(1,2,2)
hold on
plot(aa(1:100,1),aa(1:100,2)); 
plot(aa(1:100,1),aa(1:100,3)); 
plot(aa(1:100,1),aa(1:100,4)); 
plot(aa(1:100,1),aa(1:100,5)); 
% A=histogram ( Y_samples(:,1), [min(Y_samples(:,1)):max(Y_samples(:,1))], 'Normalization','probability' );
% for i = min(Y_samples(:,1)):1:max(Y_samples(:,1))
%     k1 = find(Y_samples(:,1)==i);       % 查找大于等于0小于2的元素的数组下标
%     n1 = size(k1);                     % 统计的元素的个数
%     probability(i-min(Y_samples(:,1))+1) = n1(1)/number_of_Gillespie_runs;
% end
unique_values = unique(Y_samples(:, 1));
disp(unique_values);
for i = min(Y_samples(:,1)):1:max(Y_samples(:,1))
    k1 = find(Y_samples(:,1)==i);       % 查找大于等于0小于2的元素的数组下标
    n1 = size(k1);                     % 统计的元素的个数
    probability(i-min(Y_samples(:,1))+1) = n1(1)/number_of_Gillespie_runs;
end
figure
plot(probability)



