close all; clear; clc

M = create_maze();
[goal_row, goal_col] = find(M==10);
actions = [-1 0; 0 1; 1 0; 0 -1]; % arriba, derecha, abajo, izquierda

[m, n] = size(M);
num_actions = length(actions);

theta = 1e-6;
gamma = 0.99;

% Inicialización
V = zeros(m, n);
policy = zeros(m, n);

% Algoritmo de Iteración de Valor
% Calcula la función de valor óptima V* usando la ecuación de optimalidad de Bellman:
% V*(s) = max_a [R(s,a) + gamma * V*(s')]
while true
    delta = 0;

    for i = 1 : m
        for j = 1 : n

            % Saltar paredes y meta (estados no accesibles o terminales)
            if M(i, j) == -2 || M(i, j) == 10
                continue
            end

            v = V(i, j);
            
            % Calcular valores Q para todas las acciones
            action_values = zeros(1, num_actions);

            for action = 1 : num_actions

                % Calcular siguiente estado tentativo
                next_i = i + actions(action, 1);
                next_j = j + actions(action, 2);

                % Manejo de límites y paredes: rebote con penalización
                if next_i < 1 || next_i > m || next_j < 1 || next_j > n || M(next_i, next_j) == -2
                    reward = -2;        % Penalización por chocar
                    next_i = i;         % Permanece en el mismo estado (rebote)
                    next_j = j;
                else
                    reward = M(next_i, next_j);  % Recompensa del estado destino
                end

                % Valor Q(s,a) = R + gamma * V(s')
                action_values(action) = reward + gamma * V(next_i, next_j);
            end

            % Actualización de Bellman: V(s) = max_a Q(s,a)
            max_val = max(action_values);
            V(i, j) = max_val;
            
            % Extraer política greedy (romper empates aleatoriamente)
            best_actions = find(max_val == action_values);
            policy(i, j) = best_actions(randi(length(best_actions)));

            % Actualizar métrica de convergencia
            delta = max(delta, abs(v - V(i, j)));
        end
    end

    % Criterio de parada
    if delta < theta
        break
    end
end

% Establecer valor de la meta (para visualización)
V(M==10) = 10;

% Visualización
draw_heatmap(V)

start_position = [1 2];
draw_maze(M, start_position, policy, [goal_row goal_col])