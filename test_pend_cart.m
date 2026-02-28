close all; clear, clc

M = 5;
m = 1;
L = 2;
g = 9.8;
b = 0.01;
d = 0.5;

tspan = 0:0.02:10;
x0 = [0; 0; pi+0.5; 0];

u = 0;

fun = @(t,x) pend_cart(t, x, M, m, L, g, b, d, u);

[t, x] = ode45(fun, tspan, x0);

for i = 1 : length(t)
    draw_pend_cart(x(i,:), M, m, L)
end