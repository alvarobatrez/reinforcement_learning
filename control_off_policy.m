close all; clear, clc

M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

gamma = 0.99;
epsilon = 1;
decay = 0.99;
num_episodes = 1500;
max_steps = 1e5;

Q = -100 * ones(m, n, num_actions);
Q(goal_row, goal_col, :) = 0;
Q(repmat(M==-2, 1, 1, num_actions)) = 0;
C = zeros(m, n, num_actions);
[~, pi] = max(Q, [], 3);
pi(M==-2) = 0;
pi(M==10) = 0;

mu = ones(m, n, num_actions) / num_actions;

for episode = 1 : num_episodes
    [states, actions_taken, rewards] = generate_episode(M, mu, start_position, [goal_row, goal_col], actions, num_actions, max_steps, m, n);
    G = 0;
    W = 1;

    for t = length(states) : -1 : 1
        G = gamma * G + rewards(t);

        index = sub2ind([m, n, num_actions], states(t,1), states(t,2), actions_taken(t));

        C(index) = C(index) + W;
        Q(index) = Q(index) + (W / C(index)) * (G - Q(index));

        [~, A] = max(Q(states(t, 1), states(t, 2), :));
        pi(states(t, 1), states(t, 2)) = A;

        if actions_taken(t) ~= pi(states(t, 1), states(t, 2))
            break
        end

        W = W / mu(states(t, 1), states(t, 2), actions_taken(t));
    end

    % Opcional
    epsilon = max(0.1, decay*epsilon);
    mu(states(t,1), states(t,2), :) = epsilon / num_actions;
    mu(states(t,1), states(t,2), A) = 1 - epsilon + epsilon / num_actions;

    fprintf('Episodio: %d\n', episode)
end

[~, policy] = max(Q, [], 3);
policy(M==-2) = 0;
policy(M==10) = 0;

plot_q_values(Q)

draw_maze(M, start_position, policy, [goal_row goal_col])