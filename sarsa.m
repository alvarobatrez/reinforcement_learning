close all; clear; clc
M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];
start_position = [1 2];
[goal_row, goal_col] = find(M==10);
[m, n] = size(M);
num_actions = length(actions);
alpha = 0.1;
gamma = 0.99;
epsilon = 1;
min_epsilon = 0.1;
decay = 0.99;
num_episodes = 1000;
max_steps = 1e4;
Q = zeros(m, n, num_actions);
for episode = 1 : num_episodes
    epsilon = max(min_epsilon, decay * epsilon);
    state = start_position;
    step_count = 0;
    action = egreedy_action(epsilon, Q, state, num_actions);
    while ~isequal(state, [goal_row goal_col]) && step_count < max_steps
        [next_state, reward, done] = step(M, state, action, actions, m, n);
        step_count = step_count + 1;
        if done
            Q(state(1), state(2), action) = Q(state(1), state(2), action) + alpha * (reward - Q(state(1), state(2), action));
        else
            next_action = egreedy_action(epsilon, Q, next_state, num_actions);
            Q(state(1), state(2), action) = Q(state(1), state(2), action) + alpha * (reward + gamma * Q(next_state(1), next_state(2), next_action) - Q(state(1), state(2), action));
            
            action = next_action;
        end        
        state = next_state;
    end    
    fprintf('Episodio: %d\n', episode)
end
[~, policy] = max(Q, [], 3);
policy(M==-2) = 0;
policy(M==10) = 0;

plot_q_values(Q)

draw_maze(M, start_position, policy, [goal_row goal_col])
