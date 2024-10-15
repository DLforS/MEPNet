clear;clc;
filename_10 = 'final_px_10order.txt'; 
filename_20 = 'final_px_20order.txt'; 
filename_30 = 'final_px_30order的副本.txt'; 
data_10 = readtable(filename_10, 'Format', '%f%f', 'HeaderLines', 1);
data_20 = readtable(filename_20, 'Format', '%f%f', 'HeaderLines', 1);
data_30 = load(filename_30);

final_px_10 = data_10.Var1;
final_px_20 = data_20.Var1;
final_px_30 = data_30;
gauss = data_10.Var2;
x = linspace(0,1,100)';
% Plot
figure;
hold on
plot(x,gauss, 'b-', 'LineWidth', 2,'DisplayName', 'Bimodal');
plot(x,final_px_10, 'r-.', 'LineWidth', 2,'DisplayName', '10 order');
plot(x,final_px_20, 'g-.', 'LineWidth', 2,'DisplayName', '20 order');
plot(x,final_px_30, 'm-.', 'LineWidth', 2,'DisplayName', '30 order');
xlabel('x');
ylabel('p(x)');
%title('Final PX vs. Gauss');
grid on;
set(gca, 'FontWeight', 'bold', 'FontSize', 20); 
box on; 
legend('Location', 'best'); 