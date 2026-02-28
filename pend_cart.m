function dxdt = pend_cart(t, X, M, m, L, g, b, d, u)
% M = mass cart
% m = mass pendulum
% L = length pendulum
% g = gravity
% b = friction cart
% d = friction pendulum
% u = force applied
v = X(2);
th = X(3);
w = X(4);

s = sin(th);
c = cos(th);

A = [M+m m*L*c; m*L*c m*L^2];
B = [u-b*v+m*L*w^2*s; -d*w-m*g*L*s];

x = A \ B;

dxdt = [v; x(1); w; x(2)];