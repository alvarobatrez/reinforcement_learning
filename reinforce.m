close all; clear, clc
M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];
start_position = [1 2];
[goal_row, goal_col] = find(M==10);
[m, n] = size(M);
num_actions = length(actions);
beta = 0.01;
gamma = 0.99;
num_episodes = 3000;
max_steps = 5e5;
num_envs = feature('numcores');
p = gcp('nocreate');
if isempty(p)
    parpool('local', num_envs);
else
    fprintf('Parallel environment with %d workers\n', p.NumWorkers);
end
num_inputs = 2;
layers = {{128, 'relu'} {128, 'relu'} {num_actions, 'softmax'}};
learning_rate = 0.001;
optimizer = 'adam';
loss_function = 'cross_entropy';
policy = NeuralNetwork(num_inputs, layers);
policy = policy.compile(learning_rate, optimizer, loss_function);
total_loss = zeros(num_episodes, 1);
total_returns = zeros(num_episodes, 1);
baseline = 0;
for episode = 1 : num_episodes
    grads = cell(1, num_envs);
    losses = zeros(1, num_envs);
    returns = zeros(1, num_envs);
    steps = zeros(1, num_envs);
    G0s = zeros(1, num_envs);
    parfor env = 1 : num_envs
        [s, a, r] = generate_episode_nn(M, policy, start_position, [goal_row, goal_col], actions, num_actions, max_steps, m, n);
        [grad_env, loss_env, G0_env] = episode_gradient(policy, s, a, r, gamma, beta, m, n, baseline);
        grads{env} = grad_env;
        losses(env) = loss_env;
        returns(env) = sum(r);
        steps(env) = length(a);
        G0s(env) = G0_env;
    end
    grad = grads{1};
    for env = 2 : num_envs
        for i = 1 : policy.num_layers
            grad{i} = grad{i} + grads{env}{i};
        end
    end
    for i = 1 : policy.num_layers
        grad{i} = grad{i} / num_envs;
    end
    policy = policy.update_weights(grad);
    baseline = 0.9 * baseline + 0.1 * mean(G0s);
    total_loss(episode) = mean(losses);
    total_returns(episode) = mean(returns);
    fprintf('Episodio: %d, Pasos: %.1f, Retorno: %.1f, Pérdida: %.3f\n', episode, mean(steps), total_returns(episode), total_loss(episode))
end
optimal_path = create_path(policy, M);
subplot(2,1,1), plot(1:num_episodes, total_returns), grid on
title('Retornos REINFORCE'), xlabel('Épocas'), ylabel('Retorno')
subplot(2,1,2), plot(1:num_episodes, total_loss), grid on
title('Pérdida REINFORCE'), xlabel('Épocas'), ylabel('Error')
draw_maze(M, start_position, optimal_path, [goal_row goal_col])
delete(gcp('nocreate'));