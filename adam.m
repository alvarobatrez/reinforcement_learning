function model = adam(model, grad, i)
    % Adam (Adaptive Moment Estimation)
    % Optimizador que adapta la tasa de aprendizaje por parámetro
    % usando estimaciones de primer y segundo momento del gradiente.
    %
    % Fórmulas:
    %   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t     (momento primero)
    %   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2   (momento segundo)
    %   m_hat = m_t / (1 - beta1^t)                   (corrección sesgo)
    %   v_hat = v_t / (1 - beta2^t)                   (corrección sesgo)
    %   w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
    %
    % Parámetros típicos: beta1=0.9, beta2=0.999, epsilon=1e-8
    
    model.t = model.t + 1;
    
    % Actualizar momentos
    model.m{i} = model.beta1 * model.m{i} + (1 - model.beta1) * grad{i};
    model.v{i} = model.beta2 * model.v{i} + (1 - model.beta2) * (grad{i}.^2);
    
    % Corrección de sesgo (bias correction)
    % Al inicio m y v están sesgados hacia cero, se corrige dividiendo
    m_hat = model.m{i} / (1 - model.beta1^model.t);
    v_hat = model.v{i} / (1 - model.beta2^model.t);
    
    % Actualizar pesos
    model.layers{i}.weights = model.layers{i}.weights - ...
        model.learning_rate * m_hat ./ (sqrt(v_hat) + model.epsilon);
end
