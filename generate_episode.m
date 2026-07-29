function [states, actions_taken, rewards] = generate_episode(M, policy, start_position, goal_position, actions, num_actions, max_steps, m, n)
state = start_position;
i = state(1);
j = state(2);
states = zeros(max_steps, 2);
actions_taken = zeros(max_steps, 1);
rewards = zeros(max_steps, 1);
step = 1;
while ~isequal(state, goal_position) && step <= max_steps
    states(step, :) = state;
    actions_probabilities = squeeze(policy(i, j, :));
    action = randsample(1:num_actions, 1, true, actions_probabilities);
    next_i = i + actions(action, 1);
    next_j = j + actions(action, 2);
    if next_i < 1 || next_i > m || next_j < 1 || next_j > n || M(next_i, next_j) == -2
        reward = -2;
    else
        reward = M(next_i, next_j);
        i = next_i;
        j = next_j;
    end
    state = [i j];
    actions_taken(step) = action;
    rewards(step) = reward;
    step = step + 1;
end
states = states(1 : step-1, :);
actions_taken = actions_taken(1 : step-1);
rewards = rewards(1 : step-1);
