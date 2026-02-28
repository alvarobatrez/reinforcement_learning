function y = relu(x)
    % ReLU (Rectified Linear Unit)
    % f(x) = max(0, x)
    % Derivada: f'(x) = 1 si x > 0, 0 si x < 0
    
    y = max(0, x);
end
