function [next_state, reward, done] = step(M, state, action, actions, m, n)
i = state(1);
j = state(2);
done = 0;
new_i = i + actions(action, 1);
new_j = j + actions(action, 2);

if new_i < 1 || new_i > m || new_j < 1 || new_j > n || M(new_i, new_j) == -2
    reward = -2;
    new_i = i;
    new_j = j;
else
    reward = M(new_i, new_j);
    if reward == 10
        done = 1;
    end
end

next_state = [new_i new_j];