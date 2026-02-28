close all; clear; clc

M = create_maze();
[goal_row, goal_col] = find(M==10);
actions = [-1 0; 0 1; 1 0; 0 -1]; % arriba, derecha, abajo, izquierda

[m, n] = size(M);
num_actions = length(actions);

% Inicialización: política aleatoria para estados no terminales
policy = randi(num_actions, m, n);
policy(M==-2) = 0;      % Paredes: sin política
policy(M==10) = 0;      % Meta: sin política
V = zeros(m, n);

theta = 1e-6;
gamma = 0.99;

% Algoritmo de Iteración de Políticas
% Alterna entre evaluación y mejora hasta convergencia
while true
    % Paso 1: Evaluación de la política
    V = policy_evaluation(M, policy, V, theta, gamma, actions, m, n);
    
    % Paso 2: Mejora de la política
    [V, policy, policy_stable] = policy_improvement(M, policy, V, gamma, actions, num_actions, m, n);
    
    if policy_stable
        break
    end
end

V(M==10) = 10;
draw_heatmap(V)

start_position = [1 2];
draw_maze(M, start_position, policy, [goal_row goal_col])

function V = policy_evaluation(M, pi, V, theta, gamma, actions, m, n)
    % Evalúa una política determinística dada
    % Ecuación de Bellman para política determinística:
    % V^pi(s) = R(s, pi(s)) + gamma * V^pi(s')
    
    while true
        delta = 0;

        for i = 1 : m
            for j = 1 : n
                % Saltar paredes y meta (estados terminales/absorbentes)
                if M(i, j) == -2 || M(i, j) == 10
                    continue
                end
                
                v = V(i, j);
                action = pi(i, j);
                
                % Calcular siguiente estado según la política
                next_i = i + actions(action, 1);
                next_j = j + actions(action, 2);

                % Manejo de límites y paredes: rebote con penalización
                if next_i < 1 || next_i > m || next_j < 1 || next_j > n || M(next_i, next_j) == -2
                    reward = -2;        % Penalización por chocar
                    next_i = i;         % Permanece en el mismo estado
                    next_j = j;
                else
                    reward = M(next_i, next_j);  % Recompensa del estado destino
                end

                % Actualización de Bellman: valor = recompensa + descuento * valor_siguiente
                V(i, j) = reward + gamma * V(next_i, next_j);
                delta = max(delta, abs(v - V(i,j)));
            end
        end

        % Criterio de convergencia
        if delta < theta
            break;
        end
    end
end

function [V, pi, policy_stable] = policy_improvement(M, pi, V, gamma, actions, num_actions, m, n)
    % Mejora la política de forma greedy respecto a la función de valor actual
    % pi'(s) = argmax_a [R(s,a) + gamma * V(s')]
    
    policy_stable = true;

    for i = 1 : m
        for j = 1 : n
            % Saltar paredes y meta
            if M(i, j) == -2 || M(i, j) == 10
                continue
            end

            old_action = pi(i, j);
            action_values = zeros(1, num_actions);
            
            % Evaluar todas las acciones posibles
            for action = 1 : num_actions
                next_i = i + actions(action, 1);
                next_j = j + actions(action, 2);

                % Manejo de límites y paredes
                if next_i < 1 || next_i > m || next_j < 1 || next_j > n || M(next_i, next_j) == -2
                    reward = -2;
                    next_i = i;
                    next_j = j;
                else
                    reward = M(next_i, next_j);
                end

                % Valor Q(s,a) = R + gamma * V(s')
                action_values(action) = reward + gamma * V(next_i, next_j);
            end

            % Seleccionar acción greedy (romper empates aleatoriamente)
            max_val = max(action_values);
            best_actions = find(max_val == action_values);
            pi(i, j) = best_actions(randi(length(best_actions)));

            % Verificar si la política cambió
            if old_action ~= pi(i, j)
                policy_stable = false;
            end
        end
    end
end
