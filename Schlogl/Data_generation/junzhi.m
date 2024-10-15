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
n = 100;
aa = zeros(n,4);
count = 0;
for j=0:10:50
%  for j=0.01:10:100
number_of_Gillespie_runs = n;
Y_samples = zeros(3, number_of_Gillespie_runs);
sample_time =[0,j];
for i=1:length(Y_samples)
    [t,YY] = firstReactionMethod(stoich_matrix, pfun, sample_time, x0, p);
    YY=YY';
    [m,k] = size(YY);
    Y_samples(1,i) = YY(1,k);
    Y_samples(2,i) = YY(2,k);
    Y_samples(3,i) = YY(3,k);
end
Y_samples=Y_samples';
count=count+1
% aa(count,:,:,:)=[sample_time(2),mean(Y_samples)];
aa(count,:,:,:,:,:)=[sample_time(2),mean(Y_samples)];
end
figure
subplot(1,2,1)
hold on
plot(Y_samples); 
subplot(1,2,2)
hold on
plot(aa(1:100,1),aa(1:100,2)); 
plot(aa(1:100,1),aa(1:100,3)); 
plot(aa(1:100,1),aa(1:100,4)); 
A=histogram ( Y_samples(:,2), [min(Y_samples(:,2)):max(Y_samples(:,2))], 'Normalization','probability' );
% for i = min(Y_samples(:,1)):1:max(Y_samples(:,1))
%     k1 = find(Y_samples(:,1)==i);       % 查找大于等于0小于2的元素的数组下标
%     n1 = size(k1);                     % 统计的元素的个数
%     probability(i-min(Y_samples(:,1))+1) = n1(1)/number_of_Gillespie_runs;
% end
unique_values = unique(Y_samples(:,1));
disp(unique_values);
for i = min(Y_samples(:,1)):1:max(Y_samples(:,1))
    k1 = find(Y_samples(:,1)==i);       % 查找大于等于0小于2的元素的数组下标
    n1 = size(k1);                     % 统计的元素的个数
    probability(i-min(Y_samples(:,1))+1) = n1(1)/number_of_Gillespie_runs;
end
figure
plot(probability)
