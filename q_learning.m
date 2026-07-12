close all; clear; clc

% Carga del laberinto y definición de acciones: arriba, derecha, abajo, izquierda
M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

% Parámetros del algoritmo
alpha = 0.1;        % Tasa de aprendizaje
gamma = 0.99;       % Factor de descuento
epsilon = 1;        % Parámetro de exploración inicial
decay = 0.99;       % Decaimiento de epsilon
num_episodes = 1000;
max_steps = 10000;  % Límite de pasos por episodio (evita loops infinitos)

% Inicialización de la tabla Q
Q = zeros(m, n, num_actions);

% Algoritmo Q-Learning (Off-policy TD Control)
% Actualización: Q(s,a) = Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]
% Converge a la función de valor óptima q* independientemente de la política seguida
for episode = 1 : num_episodes
    epsilon = max(0.1, decay * epsilon);
    state = start_position;
    step_count = 0;
    
    % Iterar hasta llegar a la meta o exceder el límite de pasos
    while ~isequal(state, [goal_row goal_col]) && step_count < max_steps
        % Seleccionar acción usando política epsilon-greedy (exploración)
        action = egreedy_action(epsilon, Q, state, num_actions);
        
        % Tomar acción A, observar recompensa R y siguiente estado S'
        [next_state, reward, done] = step(M, state, action, actions, m, n);
        step_count = step_count + 1;
        
        if done
            % Estado terminal: max_a' Q(s',a') = 0
            % Actualización: Q(s,a) = Q(s,a) + alpha * [R - Q(s,a)]
            Q(state(1), state(2), action) = Q(state(1), state(2), action) + ...
                alpha * (reward - Q(state(1), state(2), action));
        else
            % Q-Learning: usar max sobre todas las acciones (off-policy)
            % El valor objetivo es R + gamma * max_a' Q(s',a')
            Q(state(1), state(2), action) = Q(state(1), state(2), action) + ...
                alpha * (reward + gamma * max(Q(next_state(1), next_state(2), :)) - ...
                Q(state(1), state(2), action));
        end
        
        state = next_state;
    end
    
    fprintf('Episodio: %d\n', episode)
end

% Extraer política greedy final (óptima)
[~, policy] = max(Q, [], 3);
policy(M==-2) = 0;
policy(M==10) = 0;

plot_q_values(Q)

draw_maze(M, start_position, policy, [goal_row goal_col])
