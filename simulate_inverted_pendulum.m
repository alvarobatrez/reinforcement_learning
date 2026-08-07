function simulate_inverted_pendulum(T, X, M, m, l)
W = sqrt(M/3);
H = 0.5*W;
R = sqrt(m/6);
figure
hold on
plot([-10 10], [0 0], 'k', 'LineWidth', 2);
cart = rectangle('Position', [0 0 W H], 'Curvature', 0, 'FaceColor', 'blue', 'EdgeColor', 'k', 'LineWidth', 2);
pendulum = plot([0 0], [H W], 'k', 'LineWidth', 3);
mass = rectangle('Position', [0 0 R R], 'Curvature', 1, 'FaceColor', 'red', 'EdgeColor', 'k', 'LineWidth', 2);
axis equal
grid on
for t = 1 : length(T)
    x = X(t,1);
    th = X(t,3);
    x_cart = x - W/2;
    y_cart = 0;
    x_pend = x + l*sin(th);
    y_pend = H - l*cos(th);
    set(cart, 'Position', [x_cart y_cart W H]);
    set(pendulum, 'XData', [x x_pend], 'YData', [H y_pend]);
    set(mass, 'Position', [x_pend-R/2 y_pend-R/2 R R]);
    axis([-5 5 -2 4]);
    drawnow;
    pause(0.05);
end
hold off