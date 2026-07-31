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
    fprintf('Parallel environment with %d workers\n', p.NumWorkers);
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
        while ~isequal(state, [goal_row goal_col]) && t <= max_steps
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
        next_states = next_state(1:t, :);
        actions_taken = actions_taken(1:t);
        rewards = rewards(1:t);
        dones = dones(1:t);
        states_norm = normalize_state(states, m, n);
        next_states_norm = normalize_state(next_states, m, n);
        values = critic.predict(states_norm);
        targets = rewards + (1 - dones) .* gamma .* critic.predict(next_states_norm);
        critic_grads{env} = critic_gradient(critic, states_norm, targets);
    end
end
function grad = critic_gradient(model, state, target)
    delta = cell(1, model.num_layers);
    outputs = model.forward(state);
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
    E = size(state, 1);
    for i = 1 : model.num_layers
        grad{i} = (1 / E) * delta{i}' * [ones(E, 1), outputs{i}];
    end
end
    