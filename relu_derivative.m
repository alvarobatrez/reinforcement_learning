function y = relu_derivative(x)
    % Derivada de ReLU
    % f'(x) = 1 si x > 0
    % f'(x) = 0 si x <= 0 (incluyendo x = 0, donde técnicamente no es diferenciable)
    
    y = double(x > 0);
end
