close all; clear, clc

X = [0 0; 0 1; 1 0; 1 1];
Y = [1; 0; 0; 1];

[m, n] = size(X);
[~, num_outputs] = size(Y);

learning_rate = 0.1;
epochs = 3000;

layers = [10 10 1]; % 2 capas ocultas y 1 capa de salida

w1 = randn(layers(1), n+1);
w2 = randn(layers(2), layers(1)+1);
w3 = randn(layers(3), layers(2)+1);

total_loss = zeros(epochs, 1);

x = [ones(m, 1) X]; % sesgo

for epoch = 1 : epochs
    % Propagacion hacia adelante
    z1 = w1 * x';
    a1 = [ones(1, size(z1, 2)); sigmoid(z1)]';
    z2 = w2 * a1';
    a2 = [ones(1, size(z2, 2)); sigmoid(z2)]';
    z3 = w3 * a2';
    y_pred = sigmoid(z3)';

    % Error cuadratico medio MSE
    loss = sum((y_pred - Y).^2, 'all') / (m * num_outputs);
    total_loss(epoch) = loss;

    % Retropropagacion (gradiente descendente)
    delta3 = sigmoid_derivative(y_pred) .* (y_pred - Y);
    delta2 = sigmoid_derivative(a2(:, 2:end)) .* (delta3 * w3(:,2:end));
    delta1 = sigmoid_derivative(a1(:, 2:end)) .* (delta2 * w2(:,2:end));

    % Actualización de los pesos
    w3 = w3 - learning_rate * delta3' * a2;
    w2 = w2 - learning_rate * delta2' * a1;
    w1 = w1 - learning_rate * delta1' * x;
end

disp('Resultados:')
z1 = w1 * x';
a1 = [ones(1, size(z1, 2)); sigmoid(z1)]';
z2 = w2 * a1';
a2 = [ones(1, size(z1, 2)); sigmoid(z2)]';
z3 = w3 * a2';
y_pred = sigmoid(z3)';
for i = 1 : 4
    fprintf('Entrada: [%d %d], Salida: %.2f\n',x(i,2), x(i,3), y_pred(i))  
end

plot(1:epochs, total_loss), grid on
title('Función Costo'), xlabel('Epocas'), ylabel('Error MSE')