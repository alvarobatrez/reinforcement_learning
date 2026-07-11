close all; clear; clc

% Máscara de estados terminales (1 = terminal, 0 = no terminal).
% La terminalidad es una propiedad del estado, INDEPENDIENTE de la recompensa:
% conviene declararla de forma explícita en lugar de deducirla del valor de R.
is_terminal = [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1];

% Recompensa asociada a cada transición r(s, a, s').
% En este gridworld TODAS las transiciones desde un estado no terminal
% rinden -1 (tanto las que avanzan como las que chocan contra un borde
% o entran en un estado terminal), por eso cabe en una sola constante.
% En un MDP general la recompensa depende de la transición concreta
% r(s, a, s') y debería indexarse por estado, acción y estado destino.
recompensa = -1;

% Acciones: [arriba, derecha, abajo, izquierda]
% Representadas como [cambio_en_fila, cambio_en_columna]
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(is_terminal);
num_actions = length(actions);

% Parámetros del algoritmo
prob = 0.25;        % Probabilidad de cada acción (política aleatoria uniforme)
theta = 0.001;      % Umbral de convergencia
gamma = 0.9;        % Factor de descuento

% Inicialización de la función de valor.
% V = 0 en los estados terminales (condición de frontera del MDP).
V = zeros(m, n);

% Algoritmo de Evaluación Iterativa de Políticas.
% Ecuación de Bellman:
%   V(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [r(s,a,s') + gamma * V(s')]
% En este entorno las transiciones son DETERMINISTAS: cada acción lleva
% a un único estado s', por lo que P(s'|s,a) = 1 y el sumatorio sobre s'
% colapsa (queda solo un término por acción).
while true
    delta = 0;

    for i = 1 : m
        for j = 1 : n

            % Saltar estados terminales: su valor permanece en 0
            % (condición de frontera del MDP, no se actualiza).
            if is_terminal(i, j)
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

                % Actualización según ecuación de Bellman.
                % Como todas las transiciones desde un estado no terminal
                % valen 'recompensa', indexamos por la transición (s,a,s').
                % V(new_i, new_j) es el valor del estado siguiente (in-place).
                suma = suma + prob * (recompensa + gamma * V(new_i, new_j));
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

disp('Estados terminales (1 = terminal)')
disp(is_terminal)

disp('Función de Valor')
disp(V)
