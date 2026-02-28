close all; clear, clc

M = create_maze();
[goal_row, goal_col] = find(M==10);
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(M);
num_actions = length(actions);

theta = 1e-6;
gamma = 0.99;

V = zeros(m, n);
policy = zeros(m, n);

while true
    delta = 0;

    for i = 1 : m
        for j = 1 : n

            if M(i, j) == -2 || M(i, j) == 10
                continue
            end

            v = V(i, j);
            action_values = zeros(1, num_actions);

            for action = 1 : num_actions

                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);

                if new_i < 1 || new_i > m || new_j < 1 || new_j > n || M(new_i, new_j) == -2
                    reward = -2;
                    new_i = i;
                    new_j = j;
                else
                    reward = M(new_i, new_j);
                end

                action_values(action) = reward + gamma * V(new_i, new_j);
            end

            max_val = max(action_values);
            best_actions = find(max_val == action_values);
            V(i, j) = max_val;
            policy(i, j) = best_actions(randi(length(best_actions)));

            delta = max(delta, abs(v - V(i, j)));
        end
    end

    if delta < theta
        break
    end
end

V(M==10) = 10;
draw_heatmap(V)

start_position = [1 2];
draw_maze(M, start_position, policy, [goal_row goal_col])