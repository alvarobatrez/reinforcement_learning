close all; clear, clc

J = 3.4e-5;
b = 2.2e-5;
K = 50e-3;
L = 7.7e-3;
R = 11.4;

v_min = 0; v_max = 1;
w_min = 0; w_max = 20;
bins_v = 50;
bins_w = 50;
low = [v_min w_min];
high = [v_max w_max];
n_tilings = 4;

[tilings_v, tilings_w] = create_tilings([bins_v bins_w], low, high, n_tilings);

actions = [-0.01 0 0.01];
num_actions = length(actions);

ref = 10;
Ts = 0.01;
T = 1;
t_steps = T / Ts;

alpha = 0.1;
gamma = 0.99;
epsilon = 1;
decay = 0.99;
num_episodes = 2000;

Q = zeros(n_tilings, bins_v, bins_w, num_actions);

sum_returns = zeros(num_episodes, 1);

for episode = 1 : num_episodes
    epsilon = max(0.01, decay*epsilon);

    v = 0;
    w = 0;
    i = 0;
    state = discretize_state([v w], tilings_v, tilings_w);

    G = 0;
    
    for t = 1 : t_steps
        action = egreedy_action(epsilon, Q, state, num_actions, n_tilings);
        v = clamp_voltage(v + actions(action), v_min, v_max);

        [w, i] = simulate_motor([w i], Ts, J, b, K, L, R, v);

        next_state = discretize_state([v w], tilings_v, tilings_w);
        next_action = egreedy_action(epsilon, Q, next_state, num_actions, n_tilings);

        reward = -(ref - w)^2;

        for n = 1 : n_tilings
            Q(n, state(n, 1), state(n, 2), action) = Q(n, state(n, 1), state(n, 2), action) + ...
            alpha * (reward + gamma * Q(n, next_state(n, 1), next_state(n, 2), next_action) - Q(n, state(n, 1), state(n, 2), action));
        end

        state = next_state;
        action = next_action;

        G = G + reward;
    end

    sum_returns(episode) = G;
    
    fprintf('Episodio: %d\n', episode)
end

initial_conditions = [0 0 0]; % v w ic
draw_response(Q, actions, num_actions, initial_conditions, n_tilings, tilings_v, tilings_w, v_min, v_max, t_steps, Ts, J, b, K, L, R)

figure, plot(1:num_episodes, sum_returns)
xlabel('Episodio'), ylabel('Retornos'), title('SARSA con Agregación de Estados')

t = (0 : t_steps-1) * Ts;

function [tilings_v, tilings_w] = create_tilings(bins, low, high, n)
    tilings_v = cell(n, 1);
    tilings_w = cell(n, 1);
    
    for i = 1 : n
        low_i = low - rand * 0.2 * low;
        high_i = high + rand * 0.2 * high;
        v_centers = linspace(low_i(1), high_i(1), bins(1));
        w_centers = linspace(low_i(2), high_i(2), bins(2));
        v_edges = zeros(1, bins(1) + 1);
        v_edges(1) = low_i(1);
        v_edges(end) = high_i(1);
        
        for j = 2 : bins(1)
            v_edges(j) = (v_centers(j-1) + v_centers(j)) / 2;
        end
        
        w_edges = zeros(1, bins(2) + 1);
        w_edges(1) = low_i(2);
        w_edges(end) = high_i(2);
        
        for j = 2 : bins(2)
            w_edges(j) = (w_centers(j-1) + w_centers(j)) / 2;
        end
        
        tilings_v{i} = v_edges;
        tilings_w{i} = w_edges;
    end
end

function state = discretize_state(observations, tilings_v, tilings_w)
    v = observations(1);
    w = observations(2);
    n_tilings = length(tilings_v);
    state = zeros(n_tilings, 2);
    
    for i = 1 : n_tilings
        v_edges = tilings_v{i};
        v_idx = find(v >= v_edges(1:end-1) & v < v_edges(2:end), 1, 'first');
        
        if isempty(v_idx)
            v_idx = length(v_edges) - 1;
        end
        
        w_edges = tilings_w{i};
        w_idx = find(w >= w_edges(1:end-1) & w < w_edges(2:end), 1, 'first');
        
        if isempty(w_idx)
            w_idx = length(w_edges) - 1;
        end
        
        state(i, :) = [v_idx, w_idx];
    end
end

function action = egreedy_action(epsilon, Q, state, num_actions, n)
    if rand > epsilon
        total_action_values = zeros(1, num_actions);
        for i = 1 : n
            av = Q(i, state(i,1), state(i,2), :);
            av = reshape(av, [1, num_actions]);
            total_action_values = total_action_values + av;
        end
        [~, action] = max(total_action_values);
    else
        action = randi(num_actions);
    end
end

function v = clamp_voltage(v, v_min, v_max)
    v = max(v_min, min(v, v_max));
end

function draw_response(Q, actions, num_actions, initial_conditions, n_tilings, tilings_v, tilings_w, v_min, v_max, t_steps, Ts, J, b, K, L, R)
    v = initial_conditions(1);
    w = initial_conditions(2);
    i = initial_conditions(3);
    state = discretize_state([v w], tilings_v, tilings_w);
    vel_motor = zeros(t_steps, 1);
    
    for t = 1 : t_steps

        q = zeros(num_actions, 1);

        for a = 1 : num_actions
            for j = 1 : n_tilings
                q(a) = q(a) + Q(j, state(j,1), state(j,2), a);
            end
        end

        [~, action] = max(q);
        v = clamp_voltage(v + actions(action), v_min, v_max);
        [w, i] = simulate_motor([w i], Ts, J, b, K, L, R, v);
        state = discretize_state([v w], tilings_v, tilings_w);

        vel_motor(t) = w;
    end

    t = (0 : t_steps-1) * Ts;
    figure, plot(t, vel_motor), grid on
    xlabel('Tiempo'), ylabel('Velocidad angular'), title('Motor DC')
end