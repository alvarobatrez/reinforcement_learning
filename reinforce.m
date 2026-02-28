close all; clear, clc

M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

gamma = 0.99;
num_episodes = 1;

num_inputs = 2;
layers = {{128, 'relu'} {64, 'relu'} {4, 'softmax'}};

learning_rate = 0.001;
optimizer = 'adamW';
loss_function = 'cross_entropy';

policy = NeuralNetwork(num_inputs, layers);
policy = policy.compile(learning_rate, optimizer, loss_function);

hidden_size = layers{end-1}{1} + 1;
policy.layers{end}.weights = 0.01 * randn(num_actions, hidden_size);

total_loss = zeros(num_episodes, 1);
total_returns = zeros(num_episodes, 1);

for episode = 1 : num_episodes
    state = start_position;
    steps = 0;
    states = [];
    actions_taken = [];
    rewards = [];
    entropies = [];

    while ~isequal(state, [goal_row goal_col]) & steps < 1e5
       actions_probabilities = policy.predict(state);

       action = randsample(1:num_actions, 1, true, actions_probabilities);
       [next_state, reward, done] = step(M, state, action, actions, m, n);

       if reward == 10
           reward = 100;
       end

       states = [states; state];
       actions_taken = [actions_taken; action];
       rewards = [rewards; reward];

       state = next_state;
       steps = steps + 1;
    end

    G = zeros(size(rewards));
    ret = 0;
    
    for t = length(rewards): -1 : 1
        ret = gamma * ret + rewards(t);
        G(t) = ret;
    end
end