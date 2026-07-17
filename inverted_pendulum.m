function dx = inverted_pendulum(x, M, m, l, b, d, g, u)
s = sin(x(3));
c = cos(x(3));
v = x(2);
w = x(4);

D = M + m*s^2;

dx1 = v;
dx2 = (u -b*v + m*l*s*w^2 - m*g*s*c + d*w*c/l) / D;
dx3 = w;
dx4 = ((M + m)*g*s - (M + m)*d*w/(m*l) - u*c + b*v*c - m*l*s*c*w^2) / (l*D);

dx = [dx1; dx2; dx3; dx4];