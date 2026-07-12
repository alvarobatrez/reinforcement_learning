close all; clear; clc

% Dataset XNOR: tabla de verdad
% XNOR es 1 cuando ambas entradas son iguales
X = [0 0; 0 1; 1 0; 1 1];  % Entradas (4 muestras, 2 características)
Y = [1; 0; 0; 1];          % Salidas esperadas (XNOR)

[m, n] = size(X);          % m = 4 muestras, n = 2 entradas
[~, num_outputs] = size(Y); % num_outputs = 1

% Hiperparámetros
learning_rate = 0.1;
epochs = 3000;

% Arquitectura de la red: [10 10 1]
% Capa 1: 10 neuronas ocultas
% Capa 2: 10 neuronas ocultas  
% Capa 3: 1 neurona de salida
layers = [10 10 1];

% Inicialización de pesos con distribución normal
% w1: (10 x 3) - 10 neuronas, 2 entradas + 1 bias
% w2: (10 x 11) - 10 neuronas, 10 entradas + 1 bias
% w3: (1 x 11) - 1 neurona, 10 entradas + 1 bias
w1 = randn(layers(1), n+1);
w2 = randn(layers(2), layers(1)+1);
w3 = randn(layers(3), layers(2)+1);

total_loss = zeros(epochs, 1);

% Añadir columna de unos para el bias (término independiente)
x = [ones(m, 1) X];  % x ahora es (4 x 3)

% Entrenamiento con gradiente descendente
for epoch = 1 : epochs
    % ========== PROPAGACIÓN HACIA ADELANTE ==========
    % Capa 1: entrada -> capa oculta 1
    z1 = w1 * x';                    % (10 x 3) * (3 x 4) = (10 x 4)
    a1_sigmoid = sigmoid(z1);        % Activación sigmoid
    a1 = [ones(1, size(z1, 2)); a1_sigmoid]';  % Añadir bias: (4 x 11)
    
    % Capa 2: capa oculta 1 -> capa oculta 2
    z2 = w2 * a1';                   % (10 x 11) * (11 x 4) = (10 x 4)
    a2_sigmoid = sigmoid(z2);        % Activación sigmoid
    a2 = [ones(1, size(z2, 2)); a2_sigmoid]';  % Añadir bias: (4 x 11)
    
    % Capa 3: capa oculta 2 -> salida
    z3 = w3 * a2';                   % (1 x 11) * (11 x 4) = (1 x 4)
    y_pred = sigmoid(z3)';           % Predicción final: (4 x 1)

    % ========== CÁLCULO DE PÉRDIDA (MSE) ==========
    % Error Cuadrático Medio: L = (1/mn) * sum((y_pred - Y)^2)
    loss = sum((y_pred - Y).^2, 'all') / (m * num_outputs);
    total_loss(epoch) = loss;

    % ========== RETROPROPAGACIÓN ==========
    % Cálculo de deltas (errores) de atrás hacia adelante
    
    % Capa de salida (capa 3):
    % delta3 = dL/dy * dy/dz3 = (y_pred - Y) .* sigmoid'(z3)
    % Como sigmoid'(z) = sigmoid(z)*(1-sigmoid(z)) = y_pred*(1-y_pred)
    delta3 = sigmoid_derivative(y_pred) .* (y_pred - Y);  % (4 x 1)
    
    % Capa oculta 2:
    % delta2 = (delta3 * w3(:, sin_bias)) .* sigmoid'(z2)
    delta2 = sigmoid_derivative(a2(:, 2:end)) .* (delta3 * w3(:,2:end));  % (4 x 10)
    
    % Capa oculta 1:
    % delta1 = (delta2 * w2(:, sin_bias)) .* sigmoid'(z1)
    delta1 = sigmoid_derivative(a1(:, 2:end)) .* (delta2 * w2(:,2:end));  % (4 x 10)

    % ========== ACTUALIZACIÓN DE PESOS ==========
    % Gradiente descendente: w = w - alpha * (delta' * activacion)
    w3 = w3 - learning_rate * delta3' * a2;   % (1 x 11)
    w2 = w2 - learning_rate * delta2' * a1;   % (10 x 11)
    w1 = w1 - learning_rate * delta1' * x;    % (10 x 3)
end

% ========== EVALUACIÓN ==========
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
    fprintf('Entrada: [%d %d], Salida esperada: %d, Predicción: %.4f\n', ...
        x(i,2), x(i,3), Y(i), y_pred(i))  
end

% Gráfica de convergencia
figure;
plot(1:epochs, total_loss, 'LineWidth', 1.5), grid on
title('Función de Costo MSE'), xlabel('Épocas'), ylabel('Error')
