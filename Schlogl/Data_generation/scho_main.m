clear all, clc
x0=[250,1e5,2e5];  % Initial condition
tspan=[0:0.1:50];
[t,x]=ode23s(@(t,x) scho(t,x),tspan,x0);
figure
hold on
plot(t,x)
