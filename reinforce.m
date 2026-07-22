close all; clear, clc

M = create_medium_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

alpha = 0.0001;
beta = 0.01;
gamma = 0.99;
num_episodes = 10;
max_steps = 5e5;

num_envs = feature('numcores');
p = gcp('nocreate');
if isempty(p)
    parpool('local', num_envs);
else
    fprintf('Parallel environment with %d workers\n', p.NumWorkers);
end

num_inputs = 2;
layers = {{128, 'relu'} {64, 'relu'} {num_actions, 'softmax'}};

learning_rate = 0.001;
optimizer = 'adamW';
loss_function = 'cross_entropy';

policy = NeuralNetwork(num_inputs, layers);
policy = policy.compile(learning_rate, optimizer, loss_function);

total_loss = zeros(num_episodes, 1);
total_returns = zeros(num_episodes, 1);

states = {};
actions_taken = {};
rewards = {};

for episode = 1 : num_episodes
    parfor env = 1 : num_envs
        [s, a, r] = generate_episode_nn(M, policy, start_position, [goal_row, goal_col], actions, num_actions, max_steps, m, n);
        states{env} = s;
        actions_taken{env} = a;
        rewards{env} = r;
    end

    G = repmat({0}, 1, num_envs);
    loss = 0;
    grad = {};

    for env = 1 : num_envs
        for t = size(states{env}, 1) : -1 : 1
            G{env} = rewards{env}(t) + gamma * G{env};
            
            states_norm = normalize_state(states{env}(t,:), m, n);
            probabilities = policy.predict(states_norm);
            log_probabilities = log(probabilities + 1e-6);
            action_log_probabilities = gather_log_probs(log_probabilities, actions_taken{env}(t));

            H = -sum(probabilities .* log_probabilities);
            loss = loss + ((-gamma^t * action_log_probabilities * G{env}) - beta * H);

            grad = backpropagation(policy, grad, states_norm, actions_taken{env}(t), gamma^t * G{env}, beta);
        end
    end

    loss = loss / num_envs;

    for i = 1 : policy.num_layers
        grad{i} = grad{i} / num_envs;
    end

    policy = policy.update_weights(grad);

    g = mean(cell2mat(G));
    steps = 0;
    for i = 1 : num_envs
        steps = steps + length(states{i});
    end
    steps = round(steps / num_envs, 0);
    total_loss(episode) = loss;
    total_returns(episode) = g;

    fprintf('Episodio: %d, Pasos: %.2f, Retorno: %.2f, Pérdida: %.2f\n', episode, steps, g, loss)
end

optimal_path = create_path(policy, M, num_actions);

% Visualizar resultados
subplot(2,1,1), semilogy(1:num_episodes, total_returns), grid on
title('Retornos REINFORCE'), xlabel('Épocas'), ylabel('Retorno Promedio')
subplot(2,1,2), semilogy(1:num_episodes, total_loss), grid on
title('Pérdida REINFORCE'), xlabel('Épocas'), ylabel('Pérdida Promedio')

% Simular trayectoria óptima
draw_maze(M, start_position, optimal_path, [goal_row goal_col])

function y = gather_log_probs(log_probs, actions)
    n = size(log_probs, 1);
    indices = sub2ind(size(log_probs), (1:n)', actions);
    y = log_probs(indices);
end

function grad = compute_loss_gradients(model, state, action, weight, beta)
    delta = cell(1, model.num_layers);
    outputs = model.forward(state);

    probabilities = outputs{end};
    log_probabilities = log(probabilities + 1e-6);
    H = -sum(probabilities .* log_probabilities);

    one_hot = zeros(size(probabilities));
    one_hot(action) = 1;

    delta{end} = -weight * (one_hot - probabilities) + beta * probabilities .* (log_probabilities + H);

    for i = model.num_layers - 1 : -1 : 1
        layer_activation = model.layers{i}.activation;
        a = outputs{i+1};
        derivative = model.activation_derivative(layer_activation, a);
        w = model.layers{i + 1}.weights(:, 2:end);
        delta{i} = derivative .* (delta{i + 1} * w);
    end

    grad = cell(1, model.num_layers);
    for i = 1 : model.num_layers
        grad{i} = delta{i}' * [ones(size(state, 1), 1), outputs{i}];
    end
end

function grad = backpropagation(model, grad, state, action, weight, beta)
    new_grad = compute_loss_gradients(model, state, action, weight, beta);
    if isempty(grad)
        grad = new_grad;
    else
        for i = 1 : model.num_layers
            grad{i} = grad{i} + new_grad{i};
        end
    end
end

function optimal_path = create_path(policy, M, num_actions)
    [m, n] = size(M);
    optimal_path = zeros(m, n);
    
    for i = 1 : m
        for j = 1 : n
            if M(i, j) == -1
                state_norm = normalize_state([i j], m, n);
                actions_probabilities = policy.predict(state_norm);
                action = randsample(1:num_actions, 1, true, actions_probabilities);
    
                optimal_path(i, j) = action;
            end
           
        end
    end
end