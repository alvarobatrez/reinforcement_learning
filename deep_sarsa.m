close all; clear; clc

% Deep SARSA - Aprendizaje por Refuerzo Profundo con Redes Neuronales
% SARSA (State-Action-Reward-State-Action): Algoritmo On-policy
% Usa una red neuronal para aproximar la función Q(s,a)
% 
% Ecuación de actualización SARSA:
% Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
% donde A' se selecciona usando la misma política (epsilon-greedy)

M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];  % [arriba, derecha, abajo, izquierda]

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

% Hiperparámetros
tau = 0.005;              % Factor de actualización suave del target network (soft update)
gamma = 0.99;             % Factor de descuento
epsilon = 1;              % Parámetro de exploración inicial (epsilon-greedy)
decay = 0.995;            % Decaimiento de epsilon por episodio
num_episodes = 3000;
max_steps = 1e4;        % Límite de pasos por episodio (evita ciclos infinitos)

% Parámetros del Experience Replay (Replay de Experiencias)
buffer_capacity = 1e5;    % Capacidad máxima del buffer
batch_size = 128;         % Tamaño del batch para entrenamiento
buffer = ExperienceReplay(buffer_capacity);

% Arquitectura de la red neuronal Q
% Entrada: posición (fila, columna) -> Estado 2D
% Salida: valor Q para cada una de las 4 acciones
num_inputs = 2;
layers = {{128, 'relu'} {64, 'relu'} {num_actions, 'linear'}};  % Capa de salida lineal (Q-values)

learning_rate = 0.001;
optimizer = 'adamW';
loss_function = 'mse';

% Inicializar redes: Q-network (política actual) y Target-network (objetivo estable)
q_network = NeuralNetwork(num_inputs, layers);
q_network = q_network.compile(learning_rate, optimizer, loss_function);

target_network = NeuralNetwork(num_inputs, layers);
target_network = target_network.compile(learning_rate, optimizer, loss_function);
target_network = copy_weights(q_network, target_network);  % Copiar pesos iniciales

% Historiales para visualización
total_loss = zeros(num_episodes, 1);
total_returns = zeros(num_episodes, 1);

% Bucle principal de entrenamiento por episodios
for episode = 1 : num_episodes
    % Decaimiento de epsilon (exploración -> explotación)
    epsilon = max(0.1, decay * epsilon);
    
    state = start_position;
    steps = 0;
    loss = 0;       % Pérdida acumulada
    G = 0;          % Retorno acumulado
    n_updates = 0;  % Contador de actualizaciones de red
    
    while ~isequal(state, [goal_row goal_col]) && steps < max_steps
        steps = steps + 1;
        
        % Seleccionar acción usando política epsilon-greedy
        action = egreedy_action(epsilon, q_network, state, num_actions);
        
        % Ejecutar acción en el entorno
        [next_state, reward, done] = step(M, state, action, actions, m, n);
        
        % Recompensa shaping: refuerzo positivo al llegar a la meta
        if reward == 10
            reward = 100;
        end
        
        % Almacenar transición en el buffer de experiencias
        % Transición: (S, A, R, done, S')
        buffer = buffer.insert([state, action, reward, done, next_state]);
        
        % Entrenamiento con Experience Replay
        if buffer.can_sample(batch_size)
            % Muestrear batch aleatorio del buffer
            sample = buffer.sample(batch_size);
            [state_b, action_b, reward_b, done_b, next_state_b] = split_sample(sample);
            
            % SARSA: Seleccionar A' usando la misma política (on-policy)
            % Nota: Usamos q_network (política actual) para seleccionar la acción
            next_action_b = egreedy_action(epsilon, q_network, next_state_b, num_actions);
            
            % Evaluar Q(S', A') usando el target network (para estabilidad)
            next_q_b = gather_q(target_network, next_state_b, next_action_b, batch_size);
            
            % Calcular target: R + gamma * Q(S',A') * (1 - done)
            % Si done=1 (estado terminal), solo usamos R
            target_b = reward_b + (1 - done_b) * gamma .* next_q_b;
            
            % Obtener Q(S,A) actual para calcular el error
            current_q_b = gather_q(q_network, state_b, action_b, batch_size);
            
            % Backpropagation: actualizar Q-network para minimizar (target - Q)^2
            q_network = backpropagation(q_network, batch_size, state_b, target_b, action_b);
            
            % Soft update del target network (copia suave de pesos)
            % theta_target = tau * theta_q + (1-tau) * theta_target
            target_network = update_target_network(q_network, target_network, tau);
            
            % Actualizar métricas de pérdida (media móvil exponencial)
            n_updates = n_updates + 1;
            mse_error = mean((target_b - current_q_b).^2);
            loss = loss + (1 / n_updates) * (mse_error - loss);
        end        
        
        state = next_state;
        G = G + reward;
    end
    
    total_loss(episode) = loss;
    total_returns(episode) = G;
    fprintf('Episodio: %d, Pasos: %d, Retorno: %d, Pérdida: %.2f\n', episode, steps, G, loss)
end

% Guardar modelo entrenado
save('model_deep_sarsa.mat', 'q_network')

% Extraer política greedy final para visualización
policy = create_policy(q_network, M);

% Visualizar resultados
subplot(2,1,1), plot(1:num_episodes, total_returns), grid on
title('Retornos'), xlabel('Épocas'), ylabel('Retorno')
subplot(2,1,2), plot(1:num_episodes, total_loss), grid on
title('Pérdida'), xlabel('Épocas'), ylabel('Error MSE')

% Simular trayectoria con la política aprendida
draw_maze(M, start_position, policy, [goal_row goal_col])

%% Funciones Auxiliares

function model_copy = copy_weights(model_original, model_copy)
    % Copia los pesos de una red a otra (copia profunda)
    for i = 1 : model_original.num_layers
        model_copy.layers{i}.weights = model_original.layers{i}.weights;
    end
end

function action = egreedy_action(epsilon, model, state, num_actions)
    % Selección epsilon-greedy de acciones
    % Con probabilidad (1-epsilon): acción greedy (máximo Q)
    % Con probabilidad epsilon: acción aleatoria uniforme
    
    if rand > epsilon
        % Explotación: seleccionar acción con mayor valor Q
        % model.predict(state) retorna Q(s,a) para todas las acciones a
        [~, action] = max(model.predict(state), [], 2);
    else
        % Exploración: acción aleatoria
        [m, ~] = size(state);
        action = randi(num_actions, [m 1]);
    end
end

function [state_b, action_b, reward_b, done_b, next_state_b] = split_sample(sample)
    % Divide el batch de experiencias en sus componentes
    % Formato del buffer: [state(2), action(1), reward(1), done(1), next_state(2)]
    % Total: 7 columnas
    
    state_b = sample(:, 1:2);      % Posición actual (fila, columna)
    action_b = sample(:, 3);        % Acción tomada (1-4)
    reward_b = sample(:, 4);        % Recompensa recibida
    done_b = sample(:, 5);          % Flag de estado terminal (0 o 1)
    next_state_b = sample(:, 6:7);  % Posición siguiente (fila, columna)
end

function q = gather_q(model, state, action, batch_size)
    % Extrae los valores Q(s,a) específicos para las acciones tomadas
    % model.predict(state) retorna matriz (batch_size x num_actions)
    % Usamos sub2ind para indexar las acciones específicas
    
    q_values = model.predict(state);  % (batch_size x num_actions)
    
    % Crear índices lineales para extraer Q(s, action)
    % Índices: (fila=1:batch_size, columna=action)
    indices = sub2ind(size(q_values), (1:batch_size)', action);
    q = q_values(indices);  % Vector columna con Q(s,a) para cada muestra
end

function grad = compute_loss_gradients(model, batch_size, state, target, action)
    % Calcula gradientes para SARSA (Solo se actualiza la acción tomada)
    % La función de pérdida es MSE sobre los Q-valores de las acciones seleccionadas
    %
    % Para cada muestra i en el batch:
    % Loss_i = (target_i - Q(s_i, a_i))^2
    % donde solo la acción a_i contribuye al gradiente
    
    delta = cell(1, model.num_layers);
    outputs = model.forward(state);
    
    % Obtener predicciones actuales Q(s,a) para todas las acciones
    q_pred = outputs{end};  % (batch_size x num_actions)
    
    % Construir vector objetivo completo
    % Inicialmente igual a la predicción (gradiente cero en otras acciones)
    q_target_full = q_pred;
    
    % Modificar solo las entradas correspondientes a las acciones tomadas
    indices = sub2ind(size(q_pred), (1:batch_size)', action);
    q_target_full(indices) = target;
    
    % Gradiente del error cuadrático medio respecto a la salida
    % d(Loss)/d(Q) = 2 * (Q_pred - Q_target) / batch_size
    % Para acciones no tomadas: gradiente = 0 (porque Q_pred = Q_target)
    % Para acciones tomadas: gradiente = 2 * (Q_pred - target) / batch_size
    delta{end} = 2 * (q_pred - q_target_full) / batch_size;

    % Retropropagación del error a capas anteriores
    for i = model.num_layers - 1 : -1 : 1
        layer_activation = model.layers{i}.activation;
        a = outputs{i+1};  % Activaciones de esta capa
        
        % Derivada de la función de activación
        derivative = model.activation_derivative(layer_activation, a);
        
        % Pesos de la capa siguiente (sin término de bias)
        w = model.layers{i + 1}.weights(:, 2:end);
        
        % Regla de la cadena: delta^l = (delta^{l+1} * w^{l+1}) .* f'(a^l)
        delta{i} = derivative .* (delta{i + 1} * w);
    end

    % Calcular gradientes de los pesos
    grad = cell(1, model.num_layers);
    for i = 1 : model.num_layers
        % Gradiente respecto a pesos: (1/batch) * delta^T * [1, entrada]
        grad{i} = (1 / batch_size) * delta{i}' * [ones(batch_size, 1), outputs{i}];
    end
end

function model = backpropagation(model, batch_size, state, target, action)
    % Ejecuta un paso de backpropagation y actualización de pesos
    grad = compute_loss_gradients(model, batch_size, state, target, action);
    model = model.update_weights(grad);
end

function model_target = update_target_network(model, model_target, tau)
    % Actualización suave (soft update) del target network
    % Fórmula: theta_target <- tau * theta + (1-tau) * theta_target
    % donde tau es típicamente pequeño (ej: 0.005)
    % Esto estabiliza el entrenamiento al evitar cambios bruscos en los targets
    
    for i = 1 : model.num_layers
        model_target.layers{i}.weights = tau * model.layers{i}.weights + ...
                                         (1 - tau) * model_target.layers{i}.weights;
    end
end
