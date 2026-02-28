close all; clear; clc

% Matriz de recompensas: 0 = estados terminales, -1 = otros estados
R = [0 -1 -1 -1; -1 -1 -1 -1; -1 -1 -1 -1; -1 -1 -1 0];

% Acciones: [arriba, derecha, abajo, izquierda]
% Representadas como [cambio_en_fila, cambio_en_columna]
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(R);
num_actions = length(actions);

% Parámetros del algoritmo
prob = 0.25;        % Probabilidad de cada acción (política aleatoria uniforme)
theta = 0.001;      % Umbral de convergencia
gamma = 0.9;        % Factor de descuento

% Inicialización de la función de valor
V = zeros(m, n);

% Algoritmo de Evaluación Iterativa de Políticas
% Ecuación de Bellman: V(s) = sum_a pi(a|s) * [R(s) + gamma * V(s')]
while true
    delta = 0;

    for i = 1 : m
        for j = 1 : n

            % Saltar estados terminales (su valor permanece en 0)
            if R(i, j) == 0
                continue
            end

            v = V(i,j);
            suma = 0;
    
            for action = 1 : num_actions
    
                % Calcular nueva posición
                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);
            
                % Manejo de límites: si sale del grid, se queda en el mismo estado
                if new_i < 1 || new_i > m || new_j < 1 || new_j > n
                    new_i = i;
                    new_j = j;
                end
        
                % Actualización según ecuación de Bellman
                % R(i,j) es la recompensa por estar en el estado actual
                % V(new_i, new_j) es el valor del estado siguiente (in-place)
                suma = suma + prob * (R(i, j) + gamma * V(new_i, new_j));
            end
            
            V(i,j) = suma;
            delta = max(delta, abs(v - V(i, j)));
        end
    end

    % Criterio de parada: convergencia
    if delta < theta
        break;
    end
end

disp('Matriz de Recompensas')
disp(R)

disp('Función de Valor')
disp(V)
