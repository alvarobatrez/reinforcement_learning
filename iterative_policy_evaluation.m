close all; clear; clc

is_terminal = [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1];

% Recompensa constante: todas las transiciones valen -1
recompensa = -1;

% Acciones: [arriba, derecha, abajo, izquierda] como [delta_fila, delta_columna]
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(is_terminal);
num_actions = length(actions);

% Parámetros del algoritmo
prob = 0.25;        % Probabilidad de cada acción (política aleatoria uniforme)
theta = 0.001;      % Umbral de convergencia
gamma = 0.9;        % Factor de descuento

% Inicialización de la función de valor (0 en estados terminales)
V = zeros(m, n);

% Evaluación Iterativa de Políticas (Ecuación de Bellman).
% Transiciones deterministas: cada acción lleva a un único s'.
while true
    delta = 0;

    for i = 1 : m
        for j = 1 : n

            % Saltar estados terminales (permanecen en 0)
            if is_terminal(i, j)
                continue
            end

            v = V(i,j);
            suma = 0;

            for action = 1 : num_actions

                % Nueva posición
                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);

                % Si sale del grid, se queda en el mismo estado
                if new_i < 1 || new_i > m || new_j < 1 || new_j > n
                    new_i = i;
                    new_j = j;
                end

                % Actualización de Bellman (in-place)
                suma = suma + prob * (recompensa + gamma * V(new_i, new_j));
            end

            V(i,j) = suma;
            delta = max(delta, abs(v - V(i, j)));
        end
    end

    % Criterio de parada
    if delta < theta
        break;
    end
end

disp('Estados terminales (1 = terminal)')
disp(is_terminal)

disp('Función de Valor')
disp(V)
