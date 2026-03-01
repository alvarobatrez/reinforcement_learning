close all; clear; clc

% Deep Q-Learning (DQN) - Aprendizaje por Refuerzo Profundo
% Algoritmo Off-policy que usa una red neuronal para aproximar Q(s,a)
% Diferencia clave con SARSA: usa max Q(s',a') en lugar de Q(s',a')
% Esto lo hace más estable con Experience Replay
%
% Ecuación de actualización DQN:
% Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a' Q(S',a') - Q(S,A)]
% donde max_a' se calcula sobre todas las acciones posibles (política greedy)

M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];  % [arriba, derecha, abajo, izquierda]

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

% Hiperparámetros
tau = 0.005;              % Factor de soft update (polyak averaging)
gamma = 0.99;             % Factor de descuento
epsilon = 1;              % Exploración inicial
decay = 0.995;            % Decaimiento de epsilon
num_episodes = 3000;      
max_steps = 1e5;         % Límite de pasos por episodio

% Experience Replay
buffer_capacity = 1e5;  % Capacidad del buffer
batch_size = 128;         % Tamaño del batch
buffer = ExperienceReplay(buffer_capacity);

% Arquitectura de la red Q
num_inputs = 2;           % Estado: (fila, columna)
layers = {{128, 'relu'} {64, 'relu'} {num_actions, 'linear'}};

learning_rate = 0.001;
optimizer = 'adamW';
loss_function = 'mse';

% Inicializar redes: Q-network (online) y Target-network (estable)
q_network = NeuralNetwork(num_inputs, layers);
q_network = q_network.compile(learning_rate, optimizer, loss_function);

target_network = NeuralNetwork(num_inputs, layers);
target_network = target_network.compile(learning_rate, optimizer, loss_function);
target_network = copy_weights(q_network, target_network);

% Contadores para control de actualización del target
target_update_freq = 100;  % Actualizar target cada 100 pasos de entrenamiento
training_step = 0;         % Contador de pasos de entrenamiento

% Historiales
total_loss = zeros(num_episodes, 1);
total_returns = zeros(num_episodes, 1);

% Bucle principal de entrenamiento
for episode = 1 : num_episodes
    epsilon = max(0.1, decay * epsilon);
    state = start_position;
    steps = 0;
    loss = 0;
    G = 0;
    n_updates = 0;
    
    while ~isequal(state, [goal_row goal_col]) && steps < max_steps
        steps = steps + 1;
        
        % Seleccionar acción epsilon-greedy
        action = egreedy_action(epsilon, q_network, state, num_actions);
        
        % Ejecutar acción
        [next_state, reward, done] = step(M, state, action, actions, m, n);
        
        % Recompensa shaping
        if reward == 10
            reward = 100;
        end
        
        % Almacenar transición
        buffer = buffer.insert([state, action, reward, done, next_state]);
        
        % Entrenar si hay suficientes datos
        if buffer.can_sample(batch_size)
            sample = buffer.sample(batch_size);
            [state_b, action_b, reward_b, done_b, next_state_b] = split_sample(sample);
            
            % DQN: Calcular target usando max Q(s', a')
            % next_q_values: (batch_size x num_actions)
            next_q_values = target_network.predict(next_state_b);
            
            % max Q(s', a') sobre todas las acciones
            next_q_b = max(next_q_values, [], 2);
            
            % Target: R + gamma * max Q(s',a') * (1 - done)
            % Si done=1, el término de futuro es 0
            target_b = reward_b + (1 - done_b) * gamma .* next_q_b;
            
            % Obtener Q(s,a) actual para calcular error
            current_q_b = gather_q(q_network, state_b, action_b, batch_size);
            
            % Backpropagation
            q_network = backpropagation(q_network, batch_size, state_b, target_b, action_b);
            
            % Actualizar target network cada target_update_freq pasos
            training_step = training_step + 1;
            if mod(training_step, target_update_freq) == 0
                target_network = update_target_network(q_network, target_network, tau);
            end
            
            % Calcular pérdida (MSE)
            n_updates = n_updates + 1;
            mse_error = mean((target_b - current_q_b).^2);
            loss = loss + (mse_error - loss) / n_updates;  % Media móvil incremental
        end        
        
        state = next_state;
        G = G + reward;
    end
    
    total_loss(episode) = loss;
    total_returns(episode) = G;
    fprintf('Episodio: %d, Pasos: %d, Retorno: %d, Pérdida: %.2f\n', ...
        episode, steps, G, loss)
end

% Guardar modelo
save('model_q_learning.mat', 'q_network')

% Extraer política greedy
policy = create_policy(q_network, M);

% Visualizar resultados
subplot(2,1,1), plot(1:num_episodes, total_returns), grid on
title('Retornos DQN'), xlabel('Épocas'), ylabel('Retorno')
subplot(2,1,2), plot(1:num_episodes, total_loss), grid on
title('Pérdida DQN'), xlabel('Épocas'), ylabel('Error MSE')

% Simular trayectoria óptima
draw_maze(M, start_position, policy, [goal_row goal_col])

%% Funciones Auxiliares

function model_copy = copy_weights(model_original, model_copy)
    % Copia profunda de pesos
    for i = 1 : model_original.num_layers
        model_copy.layers{i}.weights = model_original.layers{i}.weights;
    end
end

function action = egreedy_action(epsilon, model, state, num_actions)
    % Selección epsilon-greedy
    if rand > epsilon
        [~, action] = max(model.predict(state), [], 2);
    else
        [m, ~] = size(state);
        action = randi(num_actions, [m 1]);
    end
end

function [state_b, action_b, reward_b, done_b, next_state_b] = split_sample(sample)
    % Dividir batch en componentes
    state_b = sample(:, 1:2);
    action_b = sample(:, 3);
    reward_b = sample(:, 4);
    done_b = sample(:, 5);
    next_state_b = sample(:, 6:7);
end

function q = gather_q(model, state, action, batch_size)
    % Extraer Q(s,a) para acciones específicas
    q_values = model.predict(state);
    indices = sub2ind(size(q_values), (1:batch_size)', action);
    q = q_values(indices);
end

function grad = compute_loss_gradients(model, batch_size, state, target, action)
    % Calcular gradientes para DQN
    % La función de pérdida es MSE sobre los pares (s,a) muestreados
    % Solo las acciones "action" contribuyen al gradiente
    
    delta = cell(1, model.num_layers);
    outputs = model.forward(state);
    
    % Q predichos por la red
    q_pred = outputs{end};  % (batch_size x num_actions)
    
    % Construir vector objetivo completo
    % Inicialmente igual a predicción (gradiente cero en todas las acciones)
    q_target_full = q_pred;
    
    % Actualizar solo las entradas de las acciones tomadas
    indices = sub2ind(size(q_pred), (1:batch_size)', action);
    q_target_full(indices) = target;
    
    % Gradiente del MSE: dL/dQ = 2*(Q_pred - Q_target) / batch_size
    % Para acciones no tomadas: gradiente = 0 (Q_pred = Q_target)
    % Para acciones tomadas: gradiente = 2*(Q_pred - target) / batch_size
    delta{end} = 2 * (q_pred - q_target_full) / batch_size;

    % Backpropagation
    for i = model.num_layers - 1 : -1 : 1
        layer_activation = model.layers{i}.activation;
        a = outputs{i+1};
        derivative = model.activation_derivative(layer_activation, a);
        w = model.layers{i + 1}.weights(:, 2:end);
        delta{i} = derivative .* (delta{i + 1} * w);
    end

    % Gradientes de pesos
    grad = cell(1, model.num_layers);
    for i = 1 : model.num_layers
        grad{i} = (1 / batch_size) * delta{i}' * [ones(batch_size, 1), outputs{i}];
    end
end

function model = backpropagation(model, batch_size, state, target, action)
    % Actualizar pesos
    grad = compute_loss_gradients(model, batch_size, state, target, action);
    model = model.update_weights(grad);
end

function model_target = update_target_network(model, model_target, tau)
    % Soft update (Polyak averaging)
    % theta_target = tau * theta + (1 - tau) * theta_target
    for i = 1 : model.num_layers
        model_target.layers{i}.weights = tau * model.layers{i}.weights + ...
                                         (1 - tau) * model_target.layers{i}.weights;
    end
end
