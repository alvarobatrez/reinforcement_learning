close all; clear, clc
M = create_medium_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];
start_position = [1 2];
[goal_row, goal_col] = find(M==10);
[m, n] = size(M);
num_actions = length(actions);
gamma = 0.99;
beta = 0.1;
max_grad_norm = 0.5;
min_beta = 0.01;
decay = 0.995;
num_episodes = 500;
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
actor.compile(learning_rate, optimizer, actor_loss_function);
critic = NeuralNetwork(num_inputs, critic_layers);
critic.compile(learning_rate, optimizer, critic_loss_function);
total_steps = zeros(num_episodes, 1);
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
    steps = zeros(num_envs, 1);
    H = zeros(num_envs, 1);
    for env = 1 : num_envs
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
    end
end