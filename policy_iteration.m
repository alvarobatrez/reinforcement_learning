close all; clear; clc
M = create_maze();
[goal_row, goal_col] = find(M==10);
actions = [-1 0; 0 1; 1 0; 0 -1];
[m, n] = size(M);
num_actions = length(actions);
policy = randi(num_actions, m, n);
policy(M==-2) = 0;
policy(M==10) = 0;
V = zeros(m, n);
theta = 1e-6;
gamma = 0.99;
while true
    V = policy_evaluation(M, policy, V, theta, gamma, actions, m, n);
    [V, policy, policy_stable] = policy_improvement(M, policy, V, gamma, actions, num_actions, m, n);
    if policy_stable
        break
    end
end
V(M==10) = 10;
draw_heatmap(V)
start_position = [1 2];
draw_maze(M, start_position, policy, [goal_row goal_col])
function V = policy_evaluation(M, policy, V, theta, gamma, actions, m, n)
    while true
        delta = 0;
        for i = 1 : m
            for j = 1 : n
                if M(i, j) == -2 || M(i, j) == 10
                    continue
                end
                v = V(i, j);
                action = policy(i, j);
                next_i = i + actions(action, 1);
                next_j = j + actions(action, 2);                
                if next_i < 1 || next_i > m || next_j < 1 || next_j > n || M(next_i, next_j) == -2
                    reward = -2;
                    next_i = i;
                    next_j = j;
                else
                    reward = M(next_i, next_j);
                end                
                V(i, j) = reward + gamma * V(next_i, next_j);
                delta = max(delta, abs(v - V(i,j)));
            end
        end        
        if delta < theta
            break;
        end
    end
end
function [V, policy, policy_stable] = policy_improvement(M, policy, V, gamma, actions, num_actions, m, n)
    policy_stable = true;
    for i = 1 : m
        for j = 1 : n            
            if M(i, j) == -2 || M(i, j) == 10
                continue
            end
            old_action = policy(i, j);
            action_values = zeros(1, num_actions);            
            for action = 1 : num_actions
                next_i = i + actions(action, 1);
                next_j = j + actions(action, 2);                
                if next_i < 1 || next_i > m || next_j < 1 || next_j > n || M(next_i, next_j) == -2
                    reward = -2;
                    next_i = i;
                    next_j = j;
                else
                    reward = M(next_i, next_j);
                end                
                action_values(action) = reward + gamma * V(next_i, next_j);
            end            
            max_val = max(action_values);
            best_actions = find(max_val == action_values);
            policy(i, j) = best_actions(randi(length(best_actions)));            
            if old_action ~= policy(i, j)
                policy_stable = false;
            end
        end
    end
end