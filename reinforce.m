close all; clear, clc

M = create_medium_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

alpha = 0.0001;
gamma = 0.99;
num_episodes = 1000;
max_steps = 5e3;

num_inputs = 2;
layers = {{128, 'relu'} {64, 'relu'} {num_actions, 'softmax'}};

learning_rate = 0.001;
optimizer = 'adamW';
loss_function = 'cross_entropy';

policy = NeuralNetwork(num_inputs, layers);
policy = policy.compile(learning_rate, optimizer, loss_function);