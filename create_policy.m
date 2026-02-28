function policy = create_policy(model, M)
[m, n] = size(M);
policy = zeros(m, n);

for i = 1 : m
    for j = 1 : n
        if M(i, j) == -1
            [~, action] = max(model.predict([i j]));
            policy(i, j) = action;
        end
    end
end