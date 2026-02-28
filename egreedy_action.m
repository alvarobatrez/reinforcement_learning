function action = egreedy_action(epsilon, Q, state, num_actions)
if rand > epsilon
    [~, action] = max(Q(state(1), state(2), :));
else
    action = randi(num_actions);
end