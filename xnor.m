close all; clear; clc
X = [0 0; 0 1; 1 0; 1 1];
Y = [1; 0; 0; 1];
[m, n] = size(X);
[~, num_outputs] = size(Y);
learning_rate = 0.01;
epochs = 30000;
layers = [10 10 1];
w1 = randn(layers(1), n+1);
w2 = randn(layers(2), layers(1)+1);
w3 = randn(layers(3), layers(2)+1);
total_loss = zeros(epochs, 1);
x = [ones(m, 1) X];
for epoch = 1 : epochs
    z1 = w1 * x';
    a1_sigmoid = sigmoid(z1);
    a1 = [ones(1, size(z1, 2)); a1_sigmoid]';
    z2 = w2 * a1';
    a2_sigmoid = sigmoid(z2);
    a2 = [ones(1, size(z2, 2)); a2_sigmoid]';
    z3 = w3 * a2';
    y_pred = sigmoid(z3)';
    loss = sum((y_pred - Y).^2, 'all') / (m * num_outputs);
    total_loss(epoch) = loss;
    delta3 = sigmoid_derivative(y_pred) .* (y_pred - Y);
    delta2 = sigmoid_derivative(a2(:, 2:end)) .* (delta3 * w3(:,2:end));
    delta1 = sigmoid_derivative(a1(:, 2:end)) .* (delta2 * w2(:,2:end));
    w3 = w3 - learning_rate * delta3' * a2;
    w2 = w2 - learning_rate * delta2' * a1;
    w1 = w1 - learning_rate * delta1' * x;
end
disp('Resultados del entrenamiento (XNOR):')
z1 = w1 * x';
a1_sigmoid = sigmoid(z1);
a1 = [ones(1, size(z1, 2)); a1_sigmoid]';
z2 = w2 * a1';
a2_sigmoid = sigmoid(z2);
a2 = [ones(1, size(z2, 2)); a2_sigmoid]';
z3 = w3 * a2';
y_pred = sigmoid(z3)';
for i = 1 : 4
    fprintf('Entrada: [%d %d], Salida esperada: %d, Predicción: %.4f\n', x(i,2), x(i,3), Y(i), y_pred(i))  
end
figure;
plot(1:epochs, total_loss, 'LineWidth', 1.5), grid on
title('Función de Costo MSE'), xlabel('Épocas'), ylabel('Error')