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
N = 15;

Q = zeros(m, n, num_actions);

for episode = 1 : num_episodes
    epsilon = max(0.1, decay*epsilon);
    state = start_position;
    action = egreedy_action(epsilon, Q, state, num_actions);

    states = [];
    actions_taken = [];
    rewards = [];

    states(end + 1, :) = state;
    actions_taken(end + 1) = action;

    T = inf;
    t = 0;
    
    while true
        
        if t < T
            [next_state, reward, ~] = step(M, state, action, actions, m, n);
            states(end + 1, :) = next_state;
            rewards(end + 1) = reward;

            if isequal(next_state, [goal_row goal_col])
                T = t + 1;
            else
                next_action = egreedy_action(epsilon, Q, next_state, num_actions);
                actions_taken(end + 1) = next_action;
            end

            state = next_state;
            action = next_action;
        end

        tau = t - N + 1;

        if tau >= 0
            G = 0;

            for i = tau + 1 : min(tau + N, T - 1)
                G = G + gamma^(i-tau-1) * rewards(i);
            end

            if tau + N < T
                G = G + gamma^N * Q(states(tau + N, 1), states(tau + N, 2), actions_taken(tau + N));
            end

            Q(states(tau + 1, 1), states(tau + 1, 2), actions_taken(tau + 1)) = ...
                Q(states(tau + 1, 1), states(tau + 1, 2), actions_taken(tau + 1)) + ...
                alpha * (G - Q(states(tau + 1, 1), states(tau + 1, 2), actions_taken(tau + 1)));
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