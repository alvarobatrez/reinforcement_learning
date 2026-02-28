function [w, i] = simulate_motor(state, Ts, J, b, K, L, R, v)
fun = @(t,x) motor(x, J, b, K, L, R, v);
[~, x] = ode45(fun, [0 Ts], state);
w = x(end, 1);
i = x(end, 2);