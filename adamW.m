function model = adamW(model, grad, i)
    % AdamW (Adam with Weight Decay)
    % Variante de Adam que implementa decaimiento de pesos (L2) correctamente,
    % desacoplado del gradiente adaptativo. Mejora la generalización.
    %
    % Fórmulas:
    %   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    %   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    %   m_hat = m_t / (1 - beta1^t)
    %   v_hat = v_t / (1 - beta2^t)
    %   w = w - lr * [m_hat / (sqrt(v_hat) + epsilon) + weight_decay * w]
    %
    % A diferencia de L2 regular en Adam, el decaimiento se aplica directamente
    % a los pesos, no al gradiente, lo que permite mejor regularización.
    
    model.t = model.t + 1;
    
    % Actualizar momentos (igual que Adam)
    model.m{i} = model.beta1 * model.m{i} + (1 - model.beta1) * grad{i};
    model.v{i} = model.beta2 * model.v{i} + (1 - model.beta2) * (grad{i}.^2);
    
    % Corrección de sesgo
    m_hat = model.m{i} / (1 - model.beta1^model.t);
    v_hat = model.v{i} / (1 - model.beta2^model.t);
    
    % Término de decaimiento de pesos (weight decay)
    weight_decay_term = model.weight_decay * model.layers{i}.weights;
    
    % Actualizar pesos con ambos términos: gradiente adaptativo + decaimiento
    model.layers{i}.weights = model.layers{i}.weights - ...
        model.learning_rate * (m_hat ./ (sqrt(v_hat) + model.epsilon) + weight_decay_term);
end
