classdef ExperienceReplay
    % ExperienceReplay - Implementación de Replay de Experiencias (Experience Replay)
    % Almacena transiciones (S, A, R, done, S') en un buffer circular de tamaño fijo
    % Permite muestrear batches aleatorios para romper correlaciones temporales
    % Crítico para el entrenamiento estable de redes Q profundas (DQN/SARSA)
    
    properties
        capacity        % Capacidad máxima del buffer (tamaño fijo)
        memory          % Matriz que almacena las transiciones
        position = 1    % Posición actual para insertar (índice circular)
        current_size = 0  % Tamaño actual del buffer (crece hasta capacity)
    end

    methods
        function obj = ExperienceReplay(capacity)
            % Constructor - Inicializa el buffer vacío
            % capacity: número máximo de transiciones a almacenar
            
            obj.capacity = capacity;
            % Cada transición tiene: [state(2), action(1), reward(1), done(1), next_state(2)]
            % Total: 7 valores numéricos por transición
            obj.memory = zeros(capacity, 7);
        end

        function obj = insert(obj, transition)
            % Inserta una nueva transición en el buffer (sobrescritura circular)
            % transition: vector fila de 7 elementos [s_row, s_col, a, r, done, s'_row, s'_col]
            
            % Almacenar en la posición actual
            obj.memory(obj.position, :) = transition;
            
            % Incrementar tamaño actual si no hemos llegado a capacidad máxima
            if obj.current_size < obj.capacity
                obj.current_size = obj.current_size + 1;
            end
            
            % Avanzar posición circularmente (1 -> 2 -> ... -> capacity -> 1)
            obj.position = mod(obj.position, obj.capacity) + 1;
        end

        function batch = sample(obj, batch_size)
            % Muestrea un batch aleatorio de transiciones del buffer
            % batch_size: número de transiciones a muestrear
            % batch: matriz (batch_size x 7) con las transiciones seleccionadas
            
            % Generar índices aleatorios únicos dentro del rango actual
            indices = randperm(obj.current_size, batch_size);
            
            % Extraer las transiciones correspondientes
            batch = obj.memory(indices, :);
        end

        function result = can_sample(obj, batch_size)
            % Verifica si hay suficientes experiencias para muestrear un batch
            % Requiere al menos 10 veces el tamaño del batch para estabilidad inicial
            % 
            % batch_size: tamaño del batch deseado
            % result: true si se puede muestrear, false en caso contrario
            
            result = obj.current_size >= batch_size * 10;
        end
    end
end
