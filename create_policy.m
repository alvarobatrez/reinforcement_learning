function policy = create_policy(model, M)
    [m, n] = size(M);
    policy = zeros(m, n);
    for i = 1 : m
        for j = 1 : n
            if M(i, j) == -1
                state_norm = normalize_state([i j], m, n);
                q_values = model.predict(state_norm);
                [~, action] = max(q_values);
                policy(i, j) = action;
            end
        end
    end
end
