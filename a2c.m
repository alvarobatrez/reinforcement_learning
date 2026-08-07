close all; clear, clc
M = create_medium_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];
start_position = [1 2];
[goal_row, goal_col] = find(M==10);
[m, n] = size(M);
num_actions = length(actions);
gamma = 0.99;
beta = 0.1;
min_beta = 0.01;
decay = 0.995;
num_episodes = 2000;
max_steps = 5e3;
N = 100;
num_envs = feature('numcores');
p = gcp('nocreate');
if isempty(p)
    parpool('local', num_envs);
else
    fprintf('%d ambientes paralelos\n', p.NumWorkers);
end
num_inputs = 2;
actor_layers = {{64, 'relu'} {64, 'relu'} {num_actions, 'softmax'}};
critic_layers = {{64, 'relu'} {64, 'relu'} {1, 'linear'}};
learning_rate = 0.001;
optimizer = 'adam';
actor_loss_function = 'cross_entropy';
critic_loss_function = 'mse';
actor = NeuralNetwork(num_inputs, actor_layers);
actor = actor.compile(learning_rate, optimizer, actor_loss_function);
critic = NeuralNetwork(num_inputs, critic_layers);
critic = critic.compile(learning_rate, optimizer, critic_loss_function);
total_returns = zeros(num_episodes, 1);
total_actor_loss = zeros(num_episodes, 1);
total_critic_loss = zeros(num_episodes, 1);
total_h = zeros(num_episodes, 1);
env_states = repmat(start_position, num_envs, 1);
env_counts = zeros(num_envs, 1);
for episode = 1 : num_episodes
    beta = max(min_beta, decay*beta);
    actor_grads = cell(num_envs, 1);
    critic_grads = cell(num_envs, 1);
    actor_losses = zeros(num_envs, 1);
    critic_losses = zeros(num_envs, 1);
    returns = zeros(num_envs, 1);
    H = zeros(num_envs, 1);
    parfor env = 1 : num_envs
        state = env_states(env, :);
        count = env_counts(env);
        states = zeros(N, length(start_position));
        actions_taken = zeros(N, 1);
        rewards = zeros(N, 1);
        dones = zeros(N, 1);
        for t = 1 : N
            states(t, :) = state;
            state_norm = normalize_state(state, m, n);
            actions_probabilities = actor.predict(state_norm);
            action = randsample(1:num_actions, 1, true, actions_probabilities);
            [next_state, reward, done] = step(M, state, action, actions, m, n);
            actions_taken(t) = action;
            rewards(t) = reward;
            dones(t) = done;
            if done
                state = start_position;
                count = 0;
            else
                state = next_state;
                count = count + 1;
                if count >= max_steps
                    state = start_position;
                    count = 0;
                end
            end            
        end
        env_states(env, :) = state;
        env_counts(env) = count;
        last_state_norm = normalize_state(state, m, n);
        R = critic.predict(last_state_norm);
        returns_nstep = zeros(N, 1);
        for k = N : -1 : 1
            R = rewards(k) + gamma * R * (1 - dones(k));
            returns_nstep(k) = R;
        end
        states_norm = normalize_state(states, m, n);
        values = critic.predict(states_norm);
        critic_grads{env} = critic_gradient(critic, states_norm, returns_nstep);
        critic_losses(env) = mean((returns_nstep - values).^2);
        [actor_grads{env}, actor_losses(env), H(env)] = actor_gradient(actor, states_norm, actions_taken, returns_nstep, values, beta);
        returns(env) = sum(rewards);
    end
    avg_actor_grad = average_grad(actor, actor_grads, num_envs);
    avg_critic_grad = average_grad(critic, critic_grads, num_envs);
    actor = actor.update_weights(avg_actor_grad);
    critic = critic.update_weights(avg_critic_grad);
    total_returns(episode) = mean(returns);
    total_actor_loss(episode) = mean(actor_losses);
    total_critic_loss(episode) = mean(critic_losses);
    total_h(episode) = mean(H);
    fprintf('Episodio: %d, Retorno: %.1f, Entropía: %.4f, Pérdida Actor: %.4f, Pérdida Crítico: %.4f\n', episode, total_returns(episode), total_h(episode), total_actor_loss(episode), total_critic_loss(episode))
end
optimal_path = create_path(actor, M);
subplot(2,2,1), plot(1:num_episodes, total_returns), grid on
title('Retornos'), xlabel('Épocas'), ylabel('Retorno')
subplot(2,2,2), plot(1:num_episodes, total_h), grid on
title('Entropía'), xlabel('Épocas'), ylabel('Entropía')
subplot(2,2,3), plot(1:num_episodes, total_actor_loss), grid on
title('Pérdida Actor'), xlabel('Épocas'), ylabel('Error')
subplot(2,2,4), plot(1:num_episodes, total_critic_loss), grid on
title('Pérdida Crítico'), xlabel('Épocas'), ylabel('Error')
draw_maze(M, start_position, optimal_path, [goal_row goal_col])
delete(gcp('nocreate'))
function grad = critic_gradient(model, states, target)
    T = size(states, 1);
    delta = cell(model.num_layers, 1);
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
    grad = cell(model.num_layers, 1);
    for i = 1 : model.num_layers
        grad{i} = (1 / T) * delta{i}' * [ones(T, 1), outputs{i}];
    end
end
function [grad, loss, H] = actor_gradient(model, states, actions_taken, targets, values, beta)
    T = size(states, 1);
    outputs = model.forward(states);
    probabilities = outputs{end};
    log_probabilities = log(probabilities + 1e-8);
    H = -sum(probabilities .* log_probabilities, 2);
    indices = sub2ind(size(probabilities), (1:T)', actions_taken);
    action_log_probabilities = log_probabilities(indices);
    adv = targets - values;
    adv = adv / (std(adv) + 1e-8);
    loss = sum(-adv .* action_log_probabilities - beta * H);
    one_hot = zeros(size(probabilities));
    one_hot(indices) = 1;
    delta = cell(model.num_layers, 1);
    delta{end} = -adv .* (one_hot - probabilities) + beta * probabilities .* (log_probabilities + H);
    for i = model.num_layers -1 : -1 : 1
        layer_activation = model.layers{i}.activation;
        a = outputs{i+1};
        derivative = model.activation_derivative(layer_activation, a);
        w = model.layers{i+1}.weights(:, 2:end);
        delta{i} = derivative .* (delta{i+1} * w);
    end
    grad = cell(model.num_layers, 1);
    for i = 1 : model.num_layers
        grad{i} = (1 / T) * delta{i}' * [ones(T, 1), outputs{i}];
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