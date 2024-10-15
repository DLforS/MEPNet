clear;clc;
filename_nn = 'final_px_poly.txt'; 
data_nn = readtable(filename_nn, 'Format', '%f%f', 'HeaderLines', 1);

final_px_nn = data_nn.Var1;
gauss = data_nn.Var2;
x = linspace(0,1,100)';
% Plot
figure;
hold on
plot(x,gauss, 'b-', 'LineWidth', 2,'DisplayName', 'Bimodal');
plot(x,final_px_nn, 'm-.', 'LineWidth', 2,'DisplayName', 'Polynomial');
xlabel('x');
ylabel('p(x)');
%title('Final PX vs. Gauss');
grid on;
set(gca, 'FontWeight', 'bold', 'FontSize', 20);
box on; 

legend('Location', 'best'); 