function y = softmax(x)
    % Softmax - Función de activación para clasificación multiclase
    % Convierte un vector de scores en probabilidades que suman 1
    %
    % f(x_i) = exp(x_i) / sum(exp(x_j)) para todo j
    %
    % Implementación numéricamente estable:
    % Se resta max(x) para evitar overflow en exp(x) cuando x es grande
    % exp(x - max(x)) / sum(exp(x - max(x))) = exp(x) / sum(exp(x))
    
    x_max = max(x);
    y = exp(x - x_max) ./ sum(exp(x - x_max));
end
