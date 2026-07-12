function y = sigmoid_derivative(x)
    % Derivada de la función sigmoide
    % NOTA: x debe ser el valor POST-ACTIVACIÓN: x = sigmoid(z)
    % 
    % Si x = sigmoid(z), entonces:
    % f'(z) = sigmoid(z) * (1 - sigmoid(z)) = x .* (1 - x)
    %
    % Esto es más eficiente que recalcular sigmoid(z) nuevamente
    
    y = x .* (1 - x);
end
