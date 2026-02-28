close all; clear; clc

M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1]; % arriba, derecha, abajo, izquierda

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

gamma = 0.99;
epsilon = 1;
decay = 0.99;
num_episodes = 1000;
max_steps = 1e4;

% Inicialización: política ε-soft uniforme y tablas Q y N
pi = ones(m, n, num_actions) / num_actions;  % Política estocástica uniforme
Q = zeros(m, n, num_actions);                 % Función de acción-valor
N = zeros(m, n, num_actions);                 % Contador de visitas

% Algoritmo Monte Carlo On-Policy con política ε-greedy
for episode = 1 : num_episodes
    epsilon = max(0.1, decay * epsilon);
    
    % Generar episodio siguiendo la política actual
    [states, actions_taken, rewards] = generate_episode(M, pi, start_position, [goal_row goal_col], actions, num_actions, max_steps, m, n);
    
    G = 0;
    visited = false(m, n, num_actions);
    
    % Recorrido hacia atrás del episodio (First-Visit MC)
    for t = length(states) : -1 : 1
        % Calcular retorno acumulado descontado
        G = gamma * G + rewards(t);
        
        index = sub2ind([m, n, num_actions], states(t,1), states(t,2), actions_taken(t));
        
        % First-Visit: solo actualizar si es la primera visita a (s,a) en este episodio
        if ~visited(index)
            visited(index) = true;
            
            % Actualización incremental de Q (media muestral)
            N(index) = N(index) + 1;
            Q(index) = Q(index) + (1 / N(index)) * (G - Q(index));
            
            % Mejora de política: extraer acción greedy
            max_value = max(Q(states(t, 1), states(t, 2), :));
            best_actions = find(max_value == Q(states(t, 1), states(t, 2), :));
            A = best_actions(randi(length(best_actions)));
            
            % Actualizar política a ε-greedy:
            % - Acciones no greedy: ε/|A|
            % - Acción greedy: 1 - ε + ε/|A|
            pi(states(t, 1), states(t, 2), :) = epsilon / num_actions;
            pi(states(t, 1), states(t, 2), A) = 1 - epsilon + epsilon / num_actions;
        end
    end
    
    fprintf('Episodio: %d\n', episode)
end

% Extraer política greedy determinística para visualización
[~, policy] = max(Q, [], 3);
policy(M==-2) = 0;
policy(M==10) = 0;

plot_q_values(Q)

draw_maze(M, start_position, policy, [goal_row goal_col])
