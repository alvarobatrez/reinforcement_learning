function y = sigmoid(x)
    % Función sigmoide (logística)
    % f(x) = 1 / (1 + exp(-x))
    % Mapea cualquier valor a (0, 1)
    % Usada típicamente en clasificación binaria o como activación oculta
    
    y = 1 ./ (1 + exp(-x));
end
