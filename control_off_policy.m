close all; clear, clc

% Carga del laberinto y definición de acciones: arriba, derecha, abajo, izquierda
M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

% Parámetros del algoritmo
gamma = 0.99;
epsilon = 1;
decay = 0.99;
num_episodes = 1500;
max_steps = 1e5;

% Inicialización: Q con valores pesimistas (-100), C acumulador de pesos para Weighted IS
Q = -100 * ones(m, n, num_actions);
Q(goal_row, goal_col, :) = 0;       % Valor de la meta es 0
Q(repmat(M==-2, 1, 1, num_actions)) = 0;  % Paredes tienen valor 0

C = zeros(m, n, num_actions);

% Política objetivo pi: greedy determinística respecto a Q
[~, pi] = max(Q, [], 3);
pi(M==-2) = 0;
pi(M==10) = 0;

% Política de comportamiento mu: inicialmente uniforme (ε-soft)
mu = ones(m, n, num_actions) / num_actions;

% Algoritmo Monte Carlo Off-Policy con Weighted Importance Sampling
% Aprende la política óptima pi mientras explora con mu
for episode = 1 : num_episodes
    % Generar episodio siguiendo la política de comportamiento mu
    [states, actions_taken, rewards] = generate_episode(M, mu, start_position, [goal_row goal_col], actions, num_actions, max_steps, m, n);
    G = 0;  % Retorno acumulado
    W = 1;  % Peso de importancia (importance sampling ratio)

    % Recorrido hacia atrás del episodio
    for t = length(states) : -1 : 1
        G = gamma * G + rewards(t);

        index = sub2ind([m, n, num_actions], states(t,1), states(t,2), actions_taken(t));

        % Actualización Weighted Importance Sampling: Q = Q + (W/C) * (G - Q)
        C(index) = C(index) + W;
        Q(index) = Q(index) + (W / C(index)) * (G - Q(index));

        % Actualizar política objetivo a greedy
        [~, A] = max(Q(states(t, 1), states(t, 2), :));
        pi(states(t, 1), states(t, 2)) = A;

        % Si la acción tomada no es la greedy, ratio de importancia = 0
        % (porque pi es determinística: pi(a|s) = 0 si a ≠ greedy)
        if actions_taken(t) ~= pi(states(t, 1), states(t, 2))
            break
        end

        % Actualizar peso de importancia: W = W / mu(a|s)
        W = W / mu(states(t, 1), states(t, 2), actions_taken(t));
    end

    % Opcional: hacer que mu se vuelva más greedy con el tiempo
    % Nota: t=1 y A persisten del bucle anterior (últimos valores del iterador)
    epsilon = max(0.1, decay*epsilon);
    mu(states(t,1), states(t,2), :) = epsilon / num_actions;
    mu(states(t,1), states(t,2), A) = 1 - epsilon + epsilon / num_actions;

    fprintf('Episodio: %d\n', episode)
end

% Extraer política greedy final
[~, policy] = max(Q, [], 3);
policy(M==-2) = 0;
policy(M==10) = 0;

plot_q_values(Q)

draw_maze(M, start_position, policy, [goal_row goal_col])
