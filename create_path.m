function optimal_path = create_path(policy, M)
[m, n] = size(M);
optimal_path = zeros(m, n);
for i = 1 : m
    for j = 1 : n
        if M(i, j) == -1
            state_norm = normalize_state([i j], m, n);
            actions_probabilities = policy.predict(state_norm);
            [~, action] = max(actions_probabilities);
            optimal_path(i, j) = action;
        end
    end
end