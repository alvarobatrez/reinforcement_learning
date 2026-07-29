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
decay = 0.99;
num_episodes = 1000;
N = 15;
max_steps = 10000;
Q = zeros(m, n, num_actions);
for episode = 1 : num_episodes
    epsilon = max(0.1, decay * epsilon);
    state = start_position;
    action = egreedy_action(epsilon, Q, state, num_actions);
    states = zeros(max_steps + 1, 2);
    actions_taken = zeros(max_steps + 1, 1);
    rewards = zeros(max_steps + 1, 1);
    states(1, :) = state;
    actions_taken(1) = action;    
    T = inf;
    t = 0;
    step_count = 0;    
    while step_count < max_steps
        if t < T
            [next_state, reward, done] = step(M, state, action, actions, m, n);
            states(t + 2, :) = next_state;
            rewards(t + 1) = reward;
            step_count = step_count + 1;
            if done || isequal(next_state, [goal_row goal_col])
                T = t + 1;
            else
                next_action = egreedy_action(epsilon, Q, next_state, num_actions);
                actions_taken(t + 2) = next_action;
            end            
            state = next_state;
            if ~done && T == inf
                action = next_action;
            end
        end
        tau = t - N + 1;        
        if tau >= 0
            G = 0;
            upper_bound = min(tau + N, T);
            for i = (tau + 1) : upper_bound
                G = G + gamma^(i - tau - 1) * rewards(i);
            end
            if tau + N < T
                s_idx = tau + N + 1;
                a_idx = tau + N + 1;
                G = G + gamma^N * Q(states(s_idx, 1), states(s_idx, 2), actions_taken(a_idx));
            end
            s_update = tau + 1;
            Q(states(s_update, 1), states(s_update, 2), actions_taken(s_update)) = Q(states(s_update, 1), states(s_update, 2), actions_taken(s_update)) + alpha * (G - Q(states(s_update, 1), states(s_update, 2), actions_taken(s_update)));
        end        
        t = t + 1;        
        if tau == T - 1
            break;
        end
    end    
    fprintf('Episodio: %d\n', episode)
end
[~, policy] = max(Q, [], 3);
policy(M==-2) = 0;
policy(M==10) = 0;
plot_q_values(Q)
draw_maze(M, start_position, policy, [goal_row goal_col])