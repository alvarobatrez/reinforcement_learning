close all; clear, clc

M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

alpha = 0.1;
gamma = 0.99;
epsilon = 1;
decay = 0.99;
num_episodes = 1000;

Q = zeros(m, n, num_actions);

for episode = 1 : num_episodes
    epsilon = max(0.1, decay*epsilon);
    state = start_position;
    
    while ~isequal(state, [goal_row goal_col])
        action = egreedy_action(epsilon, Q, state, num_actions);
        [next_state, reward, ~] = step(M, state, action, actions, m, n);
        next_action = egreedy_action(epsilon, Q, next_state, num_actions);
        Q(state(1), state(2), action) = Q(state(1), state(2), action) + ...
            alpha * (reward + gamma * Q(next_state(1), next_state(2), next_action) - Q(state(1), state(2), action));
        state = next_state;
        action = next_action;
    end
    
    fprintf('Episodio: %d\n', episode)
end

[~, policy] = max(Q, [], 3);
policy(M==-2) = 0;
policy(M==10) = 0;

plot_q_values(Q)

draw_maze(M, start_position, policy, [goal_row goal_col])