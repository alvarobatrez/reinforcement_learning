function [states, actions_taken, rewards] = generate_episode(M, pi, start_position, goal_position, actions, num_actions, max_steps, m, n)

state = start_position;
i = state(1);
j = state(2);
states = zeros(max_steps, 2);
actions_taken = zeros(max_steps, 1);
rewards = zeros(max_steps, 1);
step = 1;

while ~isequal(state, goal_position) && step <= max_steps
    states(step, :) = state;
    actions_probabilities = squeeze(pi(i, j, :));
    action = randsample(1:num_actions, 1, true, actions_probabilities);
    new_i = i + actions(action, 1);
    new_j = j + actions(action, 2);

    if new_i < 1 || new_i > m || new_j < 1 || new_j > n
        reward = -2;
    else
        reward = M(new_i, new_j);
        
        if reward == -1 || reward == 10
            i = new_i;
            j = new_j;
        end
    end
    
    state = [i j];
    actions_taken(step) = action;
    rewards(step) = reward;

    step = step + 1;
end

states = states(1 : step-1, :);
actions_taken = actions_taken(1 : step-1);
rewards = rewards(1:step-1);