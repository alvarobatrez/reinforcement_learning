function state_norm = normalize_state(state, m, n)
    % normalize_state - Normaliza estados [fila, columna] al rango [0, 1]
    % Escala min-max: (x - min) / (max - min)
    % Mejora la estabilidad del entrenamiento al mantener las entradas de la
    % red neuronal en un rango acotado y comparable.
    %
    % Entradas:
    %   state: vector [fila, columna] (1x2) o matriz de batch (batch x 2)
    %   m: número de filas del laberinto
    %   n: número de columnas del laberinto
    %
    % Salida:
    %   state_norm: estado(s) normalizado(s) al rango [0, 1], mismo tamaño que state

    state_norm = [(state(:, 1) - 1) / (m - 1), (state(:, 2) - 1) / (n - 1)];
end
