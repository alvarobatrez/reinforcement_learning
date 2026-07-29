function state_norm = normalize_state(state, m, n)
    state_norm = [(state(:, 1) - 1) / (m - 1), (state(:, 2) - 1) / (n - 1)];
end
