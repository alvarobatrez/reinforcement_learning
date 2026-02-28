function y = softmax(x)
x_max = max(x);
y = exp(x - x_max) ./ sum(exp(x - x_max));