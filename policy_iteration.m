close all; clear, clc

M = create_maze();
[goal_row, goal_col] = find(M==10);
actions = [-1 0; 0 1; 1 0; 0 -1]; % arriba, derecha, abajo, izquierda

[m, n] = size(M);
num_actions = length(actions);

policy = randi(num_actions, m, n);
policy(M==-2) = 0;
policy(M==10) = 0;
V = zeros(m, n);

theta = 1e-6;
gamma = 0.99;

while true
    V = policy_evaluation(M, policy, V, theta, gamma, actions, m ,n);
    [V, policy, policy_stable] = policy_improvement(M, policy, V, gamma, actions, num_actions, m, n);

    if policy_stable == true
        break
    end
end

V(M==10) = 10;
draw_heatmap(V)

start_position = [1 2];
draw_maze(M, start_position, policy, [goal_row goal_col])

function V = policy_evaluation(M, pi, V, theta, gamma, actions, m, n)
    while true
        delta = 0;

        for i = 1 : m
            for j = 1 : n
                if M(i, j) == -2 || M(i, j) == 10
                    continue
                end
                
                v = V(i, j);
                action = pi(i, j);
                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);

                if new_i < 1 || new_i > m || new_j < 1 || new_j > n || M(new_i, new_j) == -2
                    reward = -2;
                    new_i = i;
                    new_j = j;
                else
                    reward = M(new_i, new_j);
                end

                V(i, j) = reward + gamma * V(new_i, new_j);
                delta = max(delta, abs(v - V(i,j)));
            end
        end

        if delta < theta
            break;
        end
    end
end

function [V, pi, policy_stable] = policy_improvement(M, pi, V, gamma, actions, num_actions, m, n)
    policy_stable = true;

    for i = 1 : m
        for j = 1 : n
            if M(i, j) == -2 || M(i, j) == 10
                continue
            end

            old_action = pi(i, j);
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
            pi(i, j) = best_actions(randi(length(best_actions)));

            if old_action ~= pi(i, j)
                policy_stable = false;
            end
        end
    end
end