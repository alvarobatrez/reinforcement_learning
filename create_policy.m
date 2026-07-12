function policy = create_policy(model, M)
    % create_policy - Extrae la política greedy determinística de una red Q entrenada
    % Para cada estado (celda) del laberinto, selecciona la acción con mayor valor Q
    %
    % Entradas:
    %   model: red neuronal entrenada (NeuralNetwork) que aproxima Q(s,a)
    %   M: matriz del laberinto (valores -1 = camino, -2 = pared, 10 = meta)
    %
    % Salida:
    %   policy: matriz (m x n) con la acción greedy (1-4) para cada estado
    %           0 en paredes y meta (estados no transitables o terminales)
    
    [m, n] = size(M);
    policy = zeros(m, n);
    
    % Iterar sobre todas las celdas del laberinto
    for i = 1 : m
        for j = 1 : n
            % Solo considerar celdas transitables (caminos libres)
            % M(i,j) == -1 indica un camino libre en la convención del laberinto
            if M(i, j) == -1
                % Predecir valores Q para el estado (i, j)
                % model.predict retorna un vector con Q(s,a) para a = 1,2,3,4
                q_values = model.predict([i j]);
                
                % Seleccionar acción greedy (la de mayor valor Q)
                [~, action] = max(q_values);
                
                % Almacenar en la matriz de política
                policy(i, j) = action;
            end
            % Nota: Las paredes (-2) y la meta (10) quedan con valor 0
        end
    end
end
