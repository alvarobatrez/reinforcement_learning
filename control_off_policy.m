close all; clear; clc

M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1]; % arriba, derecha, abajo, izquierda

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

gamma = 0.99;       % Factor de descuento
epsilon = 1;        % Parámetro de exploración inicial
decay = 0.99;       % Decaimiento de epsilon
num_episodes = 1500;
max_steps = 1e5;

% Inicialización: valores pesimistas (-100) para estimación conservadora
Q = -100 * ones(m, n, num_actions);
Q(goal_row, goal_col, :) = 0;       % Valor de la meta es 0
Q(repmat(M==-2, 1, 1, num_actions)) = 0;  % Paredes tienen valor 0

C = zeros(m, n, num_actions);       % Acumulador de pesos para Weighted IS

% Política objetivo: greedy determinística respecto a Q
[~, pi] = max(Q, [], 3);
pi(M==-2) = 0;
pi(M==10) = 0;

% Política de comportamiento: ε-soft (exploratoria)
mu = ones(m, n, num_actions) / num_actions;

% Algoritmo Monte Carlo Off-Policy con Weighted Importance Sampling
% Aprende la política óptima pi mientras explora con mu
for episode = 1 : num_episodes
    % Generar episodio siguiendo la política de comportamiento mu
    [states, actions_taken, rewards] = generate_episode(M, mu, start_position, [goal_row goal_col], actions, num_actions, max_steps, m, n);
    
    G = 0;      % Retorno acumulado
    W = 1;      % Peso de importancia (importance sampling ratio)
    
    % Recorrido hacia atrás del episodio
    for t = length(states) : -1 : 1
        % Calcular retorno descontado
        G = gamma * G + rewards(t);
        
        index = sub2ind([m, n, num_actions], states(t,1), states(t,2), actions_taken(t));
        
        % Actualización Weighted Importance Sampling:
        % Q(s,a) = Q(s,a) + (W/C) * [G - Q(s,a)]
        C(index) = C(index) + W;
        Q(index) = Q(index) + (W / C(index)) * (G - Q(index));
        
        % Actualizar política objetivo a greedy
        [~, A] = max(Q(states(t, 1), states(t, 2), :));
        pi(states(t, 1), states(t, 2)) = A;
        
        % Si la acción tomada no es la greedy, ratio de importancia = 0
        % (porque pi es determinística: pi(a|s) = 0 si a ≠ greedy)
        if actions_taken(t) ~= pi(states(t, 1), states(t, 2))
            break;
        end
        
        % Actualizar peso de importancia: W = W * pi(a|s) / mu(a|s)
        % Como pi es determinística y greedy: pi(a|s) = 1
        % Por tanto: W = W / mu(a|s)
        W = W / mu(states(t, 1), states(t, 2), actions_taken(t));
    end
    
    % Opcional: hacer que mu se vuelva más greedy con el tiempo
    % Actualizar mu para todos los estados visitados en este episodio
    epsilon = max(0.1, decay * epsilon);
    
    for t = 1 : length(states)
        if M(states(t,1), states(t,2)) ~= -2 && M(states(t,1), states(t,2)) ~= 10
            [~, A] = max(Q(states(t, 1), states(t, 2), :));
            mu(states(t,1), states(t,2), :) = epsilon / num_actions;
            mu(states(t,1), states(t,2), A) = 1 - epsilon + epsilon / num_actions;
        end
    end
    
    fprintf('Episodio: %d\n', episode)
end

% Extraer política greedy final
[~, policy] = max(Q, [], 3);
policy(M==-2) = 0;
policy(M==10) = 0;

plot_q_values(Q)

draw_maze(M, start_position, policy, [goal_row goal_col])
