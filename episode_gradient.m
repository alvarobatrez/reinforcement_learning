function [grad, loss, G0, H] = episode_gradient(model, states, actions_taken, rewards, gamma, beta, m, n, baseline)
    T = size(states, 1);

    G = zeros(T, 1);
    G(T) = rewards(T);
    for t = T - 1 : -1 : 1
        G(t) = rewards(t) + gamma * G(t + 1);
    end
    G0 = G(1);

    outputs = model.forward(normalize_state(states, m, n));
    probabilities = outputs{end};
    log_probabilities = log(probabilities + 1e-6);
    H = -sum(probabilities .* log_probabilities, 2);

    indices = sub2ind(size(probabilities), (1:T)', actions_taken);
    action_log_probabilities = log_probabilities(indices);

    A = G - baseline;
    A = A / (std(A) + 1e-8);

    weight = gamma.^((0:T - 1)') .* A;
    loss = sum(-weight .* action_log_probabilities - beta * H);

    one_hot = zeros(size(probabilities));
    one_hot(indices) = 1;

    delta = cell(1, model.num_layers);
    delta{end} = -weight .* (one_hot - probabilities) + beta * probabilities .* (log_probabilities + H);

    for i = model.num_layers - 1 : -1 : 1
        layer_activation = model.layers{i}.activation;
        a = outputs{i+1};
        derivative = model.activation_derivative(layer_activation, a);
        w = model.layers{i + 1}.weights(:, 2:end);
        delta{i} = derivative .* (delta{i + 1} * w);
    end

    grad = cell(1, model.num_layers);
    for i = 1 : model.num_layers
        grad{i} = delta{i}' * [ones(T, 1), outputs{i}] / T;
    end
    
    loss = loss / T;
    H = mean(H);
end
