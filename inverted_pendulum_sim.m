close all; clear, clc
M = 5;
m = 1;
l = 2;
b = 0.1;
d = 0.1;
g = -9.81;
u = 0;
tspan = [0 10];
x0 = [0; 0; pi-0.5; 0];
[t,y] = ode45(@(t,x) inverted_pendulum(x, M, m, l, b, d, g, u), tspan, x0);
simulate_inverted_pendulum(t, y, M, m, l)