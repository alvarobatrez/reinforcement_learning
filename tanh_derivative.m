function y = tanh_derivative(x)
    % Derivada de la función tanh (tangente hiperbólica)
    % NOTA: x debe ser el valor POST-ACTIVACIÓN: x = tanh(z)
    %
    % Si x = tanh(z), entonces:
    % f'(z) = 1 - tanh²(z) = 1 - x.^2
    %
    % La función tanh mapea a (-1, 1) y es útil para normalizar salidas
    
    y = 1 - x.^2;
end
