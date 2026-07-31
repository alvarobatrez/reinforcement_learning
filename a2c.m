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
        
    end
end