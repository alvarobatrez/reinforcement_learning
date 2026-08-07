function dx = motor(x, J, b, K, L, R, v)
dx1 = -b/J*x(1) + K/J*x(2);
dx2 = -K/L*x(1) - R/L*x(2) + v/L;
dx = [dx1; dx2];