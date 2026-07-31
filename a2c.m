close all; clear, clc
M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];
start_position = [1 2];
[goal_row, goal_col] = find(M==10);
[m, n] = size(M);
num_actions = length(actions);
gamma = 0.999;
beta = 0.1;
min_beta = 0.01;
decay = 0.995;
num_episodes = 20000;
max_steps = 5e3;
num_envs = feature('numcores');
p = gcp('nocreate');
if isempty(p)
    parpool('local', num_envs);
else
    fprintf('Ambientes paralelos con %d workers\n', p.NumWorkers);
end
num_inputs = 2;
actor_layers = {{128, 'relu'} {128, 'relu'} {num_actions, 'softmax'}};
critic_layers = {{128, 'relu'} {128, 'relu'} {1, 'linear'}};
learning_rate = 0.001;
optimizer = 'adam';
actor_loss_function = 'cross_entropy';
critic_loss_function = 'mse';
actor = NeuralNetwork(num_inputs, actor_layers);
actor = actor.compile(learning_rate, optimizer, actor_loss_function);
critic = NeuralNetwork(num_inputs, critic_layers);
critic = critic.compile(learning_rate, optimizer, critic_loss_function);
total_steps = zeros(num_episodes, 1);
total_returns = zeros(num_episodes, 1);
total_actor_loss = zeros(num_episodes, 1);
total_critic_loss = zeros(num_episodes, 1);
total_h = zeros(num_episodes, 1);
for episode = 1 : num_episodes
    beta = max(min_beta, decay * beta);
    actor_grads = cell(1, num_envs);
    critic_grads = cell(1, num_envs);
    actor_losses = zeros(1, num_envs);
    critic_losses = zeros(1, num_envs);
    returns = zeros(1, num_envs);
    steps = zeros(1, num_envs);
    H = zeros(1, num_envs);
    parfor env = 1 : num_envs
        state = start_position;
        states = zeros(max_steps, length(start_position));
        next_states = zeros(max_steps, length(start_position));
        actions_taken = zeros(max_steps, 1);
        rewards = zeros(max_steps, 1);
        dones = zeros(max_steps, 1);
        t = 0;
        while ~isequal(state, [goal_row goal_col]) && t < max_steps
            t = t + 1;
            states(t, :) = state;
            state_norm = normalize_state(state, m, n);
            actions_probabilities = actor.predict(state_norm);
            action = randsample(1:num_actions, 1, true, actions_probabilities);
            [next_state, reward, done] = step(M, state, action, actions, m, n);
            next_states(t, :) = next_state;
            actions_taken(t) = action;
            rewards(t) = reward;
            dones(t) = done;
            state = next_state;
        end
        states = states(1:t, :);
        next_states = next_states(1:t, :);
        actions_taken = actions_taken(1:t);
        rewards = rewards(1:t);
        dones = dones(1:t);
        G = zeros(t, 1);
        G(t) = rewards(t);
        for k = t - 1 : -1 : 1
            G(k) = rewards(k) + gamma * G(k + 1);
        end
        states_norm = normalize_state(states, m, n);
        values = critic.predict(states_norm);
        targets = G;
        critic_grads{env} = critic_gradient(critic, states_norm, targets);
        critic_losses(env) = mean((targets - values).^2);
        [actor_grads{env}, actor_losses(env), H(env)] = actor_gradient(actor, states_norm, actions_taken, targets, values, beta);
        returns(env) = sum(rewards);
        steps(env) = t;
    end
    avg_actor_grad = average_grad(actor, actor_grads, num_envs);
    avg_critic_grad = average_grad(critic, critic_grads, num_envs);
    actor = actor.update_weights(avg_actor_grad);
    critic = critic.update_weights(avg_critic_grad);
    mean_steps = mean(steps);
    total_steps(episode) = mean_steps;
    total_returns(episode) = mean(returns);
    total_actor_loss(episode) = mean(actor_losses);
    total_critic_loss(episode) = mean(critic_losses);
    total_h(episode) = mean(H);
    fprintf('Episodio: %d, Pasos: %.1f, Retorno: %.1f, Pérdida Actor: %.4f, Pérdida Crítico: %.4f, Entropía: %.4f\n', episode, mean_steps, total_returns(episode), total_actor_loss(episode), total_critic_loss(episode), total_h(episode))
end
delete(gcp('nocreate'));
optimal_path = create_path(actor, M);
subplot(3,2,1), semilogy(1:num_episodes, total_steps), grid on
title('Pasos'), xlabel('Épocas'), ylabel('Num Pasos')
subplot(3,2,2), plot(1:num_episodes, total_returns), grid on
title('Retornos'), xlabel('Épocas'), ylabel('Retorno')
subplot(3,2,3), plot(1:num_episodes, total_actor_loss), grid on
title('Pérdida Actor'), xlabel('Épocas'), ylabel('Error')
subplot(3,2,4), plot(1:num_episodes, total_critic_loss), grid on
title('Pérdida Crítico'), xlabel('Épocas'), ylabel('Error')
subplot(3,2,5), plot(1:num_episodes, total_h), grid on
title('Entropía'), xlabel('Épocas'), ylabel('Entropía')
draw_maze(M, start_position, optimal_path, [goal_row goal_col])
function grad = critic_gradient(model, states, target)
    T = size(states, 1);
    delta = cell(1, model.num_layers);
    outputs = model.forward(states);
    y_pred = outputs{end};
    delta{end} = y_pred - target;
    for i = model.num_layers -1 : -1 : 1
        layer_activation = model.layers{i}.activation;
        a = outputs{i+1};
        derivative = model.activation_derivative(layer_activation, a);
        w = model.layers{i+1}.weights(:, 2:end);
        delta{i} = derivative .* (delta{i+1} * w);
    end
    grad = cell(1, model.num_layers);
    for i = 1 : model.num_layers
        grad{i} = (1 / T) * delta{i}' * [ones(T, 1), outputs{i}];
    end
end
function [grad, loss, H] = actor_gradient(model, states, actions_taken, targets, values, beta)
    T = size(states, 1);
    outputs = model.forward(states);
    probabilities = outputs{end};
    log_probabilities = log(probabilities + 1e-6);
    H = -sum(probabilities .* log_probabilities, 2);
    indices = sub2ind(size(probabilities), (1:T)', actions_taken);
    action_log_probabilities = log_probabilities(indices);
    adv = targets - values;
    adv = adv / (std(adv) + 1e-8);
    loss = sum(-adv .* action_log_probabilities - beta * H);
    one_hot = zeros(size(probabilities));
    one_hot(indices) = 1;
    delta = cell(1, model.num_layers);
    delta{end} = -adv .* (one_hot - probabilities) + beta * probabilities .* (log_probabilities + H);
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
function grad = average_grad(model, grads, num_envs)
    grad = grads{1};
    for env = 2 : num_envs
        for i = 1 : model.num_layers
            grad{i} = grad{i} + grads{env}{i};
        end
    end
    for i = 1 : model.num_layers
        grad{i} = grad{i} / num_envs;
    end
end   