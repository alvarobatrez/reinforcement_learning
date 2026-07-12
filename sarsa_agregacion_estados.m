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
range_v = linspace(v_min, v_max, bins_v);
range_w = linspace(w_min, w_max, bins_w);

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

Q = zeros(bins_v, bins_w, num_actions);

sum_returns = zeros(num_episodes, 1);

for episode = 1 : num_episodes
    epsilon = max(0.01, decay*epsilon);

    v = 0;
    w = 0;
    i = 0;
    state = discretize_state([v w], range_v, range_w);

    G = 0;
    
    for t = 1 : t_steps
        action = egreedy_action(epsilon, Q, state, num_actions);
        v = clamp_voltage(v + actions(action), v_min, v_max);

        [w, i] = simulate_motor([w i], Ts, J, b, K, L, R, v);

        next_state = discretize_state([v w], range_v, range_w);
        next_action = egreedy_action(epsilon, Q, next_state, num_actions);

        reward = -(ref - w)^2;

        Q(state(1), state(2), action) = Q(state(1), state(2), action) + ...
            alpha * (reward + gamma * Q(next_state(1), next_state(2), next_action) - Q(state(1), state(2), action));

        state = next_state;
        action = next_action;

        G = G + reward;
    end

    sum_returns(episode) = G;
    
    fprintf('Episodio: %d\n', episode)
end

[~, policy] = max(Q, [], 3);

initial_conditions = [0 0 0]; % v w i
draw_response(policy, actions, initial_conditions, range_v, range_w, v_min, v_max, t_steps, Ts, J, b, K, L, R)

figure, plot(1:num_episodes, sum_returns)
xlabel('Episodio'), ylabel('Retornos'), title('SARSA con Agregación de Estados')

t = (0 : t_steps-1) * Ts;

function state = discretize_state(observations, range_v, range_w)
    v = observations(1);
    w = observations(2);
    state(1) = discretize(v, range_v, 'IncludedEdge', 'left');
    state(2) = discretize(w, range_w, 'IncludedEdge', 'left');
end

function v = clamp_voltage(v, v_min, v_max)
    v = max(v_min, min(v, v_max));
end

function draw_response(policy, actions, initial_conditions, range_v, range_w, v_min, v_max, t_steps, Ts, J, b, K, L, R)
    v = initial_conditions(1);
    w = initial_conditions(2);
    i = initial_conditions(3);

    state = discretize_state([v w], range_v, range_w);

    vel_motor = zeros(t_steps, 1);
    
    for t = 1 : t_steps
        action = policy(state(1), state(2));
        v = clamp_voltage(v + actions(action), v_min, v_max);
        [w, i] = simulate_motor([w i], Ts, J, b, K, L, R, v);
        state = discretize_state([v w], range_v, range_w);

        vel_motor(t) = w;
    end

    t = (0 : t_steps-1) * Ts;
    figure, plot(t, vel_motor), grid on
    xlabel('Tiempo'), ylabel('Velocidad angular'), title('Motor DC')
end