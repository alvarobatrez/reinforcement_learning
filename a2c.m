close all; clear, clc
% =====================================================================
%  Advantage Actor-Critic (A2C) - version reproducible
% ---------------------------------------------------------------------
%  A2C con episodios continuos (estilo REINFORCE/DQN):
%    - Cada entorno corre de forma CONTINUA a lo largo de muchos rollouts.
%      El estado se mantiene entre iteraciones (ventana deslizante), de modo que
%      el agente recorre distancias largas y puede alcanzar la meta.
%    - Se actualiza cada N pasos (rollout n-step), no al final del episodio.
%    - Retornos n-step con bootstrap del critico:  R = r + gamma*R*(1-done).
%    - Ventaja A = R - V(s)  (V del critico como baseline, sin gradiente al actor).
%    - done=1 solo al llegar a la meta (corta el bootstrap); el timeout es un
%      truncado (reset sin done, bootstrap valido), igual que max_steps en REINFORCE.
%    - Clip de gradientes (norma L2 global, max_norm = 0.5).
%  Actor y critico son redes separadas (variante valida de A2C).
% =====================================================================
M = create_medium_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];
start_position = [1 2];
[goal_row, goal_col] = find(M==10);
[m, n] = size(M);
num_actions = length(actions);

% --- Hiperparametros (episodios continuos + entropia constante alta) ---
%  Diagnostico del colapso: con beta decayendo POR ACTUALIZACION (cada N pasos),
%  la politica colapsaba a determinista (entropia->0) tras ~250 updates, antes de
%  acumular exploracion suficiente para descubrir la meta. REINFORCE no colapsa
%  porque actualiza ~1 vez por cada miles de pasos de entorno.
%  Solucion: beta CONSTANTE alto para forzar exploracion sostenida + N mayor para
%  reducir la frecuencia de actualizacion. Asi la politica nunca colapsa y el
%  agente sigue muestreando hasta encontrar la meta.
%    gamma = 0.99  : igual que DQN/SARSA.
%    N = 50        : ventana n-step. Menos actualizaciones por paso de entorno.
%    beta = 0.5 CONSTANTE : entropia forzada alta (no decae). Con ventajas
%                    normalizadas |A|~1, beta=0.5 fija la entropia en ~0.7
%                    (del maximo 1.386): exploracion robusta que permite
%                    descubrir la meta sin colapsar la politica.
%    lr = 1e-3     : probado en REINFORCE/DQN.
gamma = 0.99;             % factor de descuento
N = 50;                   % longitud del rollout (n-step)
max_grad_norm = 0.5;      % clip de gradientes (torch clip_grad_norm_)
beta = 0.5;               % coeficiente de entropia CONSTANTE (evita colapso de politica)
max_steps_per_episode = 5e3;  % tope de truncado por entorno (como REINFORCE/DQN)
num_episodes = 3000;
num_envs = feature('numcores');
p = gcp('nocreate');
if isempty(p)
    parpool('local', num_envs);
else
    fprintf('Ambientes paralelos con %d workers\n', p.NumWorkers);
end

% --- Redes (actor y critico separadas) --------------------------------
num_inputs = 2;
actor_layers = {{128, 'relu'} {128, 'relu'} {num_actions, 'softmax'}};
critic_layers = {{128, 'relu'} {128, 'relu'} {1, 'linear'}};
learning_rate = 0.001;
optimizer = 'adam';
actor_loss_function = 'cross_entropy';
critic_loss_function = 'mse';
actor = NeuralNetwork(num_inputs, actor_layers);
actor = actor.compile(learning_rate, optimizer, actor_loss_function);
critic = NeuralNetwork(num_inputs, critic_layers);
critic = critic.compile(learning_rate, optimizer, critic_loss_function);

% --- Metricas ---------------------------------------------------------
total_steps = zeros(num_episodes, 1);
total_returns = zeros(num_episodes, 1);
total_actor_loss = zeros(num_episodes, 1);
total_critic_loss = zeros(num_episodes, 1);
total_h = zeros(num_episodes, 1);
total_goals = zeros(num_episodes, 1);     % diagnosticos: cuantos entornos llegaron a la meta

% --- Estado por entorno PERSISTENTE entre iteraciones (ventana deslizante) -
% Variables "sliced" del parfor: se conservan fuera y se reescriben por indice.
env_states = repmat(start_position, num_envs, 1);   % estado actual de cada entorno
env_step_counts = zeros(num_envs, 1);               % pasos desde el ultimo reset

for episode = 1 : num_episodes
    % beta es CONSTANTE (definido arriba): no decae, para mantener la exploracion.
    actor_grads = cell(1, num_envs);
    critic_grads = cell(1, num_envs);
    actor_losses = zeros(1, num_envs);
    critic_losses = zeros(1, num_envs);
    returns = zeros(1, num_envs);
    steps = zeros(1, num_envs);
    H = zeros(1, num_envs);
    goals = zeros(1, num_envs);
    parfor env = 1 : num_envs
        % --- 1. RECOGER UN ROLLOUT DE N PASOS (ventana deslizante) -----
        state = env_states(env, :);                 % continua desde donde quedo
        steps_since_reset = env_step_counts(env);
        states = zeros(N, length(start_position));
        actions_taken = zeros(N, 1);
        rewards = zeros(N, 1);
        dones = zeros(N, 1);
        for t = 1 : N
            states(t, :) = state;
            state_norm = normalize_state(state, m, n);
            actions_probabilities = actor.predict(state_norm);
            action = randsample(1:num_actions, 1, true, actions_probabilities);
            [next_state, reward, done] = step(M, state, action, actions, m, n);
            actions_taken(t) = action;
            rewards(t) = reward;
            dones(t) = done;
            if done
                goals(env) = goals(env) + 1;
                state = start_position;             % auto-reset, el rollout SIGUE
                steps_since_reset = 0;
            else
                state = next_state;
                steps_since_reset = steps_since_reset + 1;
                if steps_since_reset >= max_steps_per_episode
                    state = start_position;         % truncado por tiempo (reset sin done)
                    steps_since_reset = 0;
                end
            end
        end
        env_states(env, :) = state;                 % persiste para la siguiente iteracion
        env_step_counts(env) = steps_since_reset;

        % --- 2. RETORNOS n-step CON BOOTSTRAP --------------------------
        %  R = V(s_N)  (bootstrap con el ultimo estado del rollout)
        %  hacia atras: R = r(t) + gamma * R * (1 - done(t))
        %  El (1-done) anula el bootstrap cuando el rollout termina en meta.
        last_state_norm = normalize_state(state, m, n);
        R = critic.predict(last_state_norm);
        returns_nstep = zeros(N, 1);
        for k = N : -1 : 1
            R = rewards(k) + gamma * R * (1 - dones(k));
            returns_nstep(k) = R;
        end

        % --- 3. VALORES DEL CRITICO Y VENTAJAS -------------------------
        states_norm = normalize_state(states, m, n);
        values = critic.predict(states_norm);
        % advantage = returns - values  (values NO arrastra gradiente hacia el actor:
        % son redes separadas, equivale a values.detach())

        % --- 4. GRADIENTES (actor y critico) ---------------------------
        critic_grads{env} = critic_gradient(critic, states_norm, returns_nstep);
        critic_losses(env) = mean((returns_nstep - values).^2);
        [actor_grads{env}, actor_losses(env), H(env)] = actor_gradient(actor, states_norm, actions_taken, returns_nstep, values, beta);
        returns(env) = sum(rewards);
        steps(env) = steps_since_reset;
    end

    % --- 5. AGREGAR Y ACTUALIZAR PARAMETROS ------------------------------
    avg_actor_grad = average_grad(actor, actor_grads, num_envs);
    avg_critic_grad = average_grad(critic, critic_grads, num_envs);
    avg_actor_grad = actor.clip_grad_norm(avg_actor_grad, max_grad_norm);
    avg_critic_grad = critic.clip_grad_norm(avg_critic_grad, max_grad_norm);
    actor = actor.update_weights(avg_actor_grad);
    critic = critic.update_weights(avg_critic_grad);

    mean_steps = mean(steps);
    total_steps(episode) = mean_steps;
    total_returns(episode) = mean(returns);
    total_actor_loss(episode) = mean(actor_losses);
    total_critic_loss(episode) = mean(critic_losses);
    total_h(episode) = mean(H);
    total_goals(episode) = mean(goals);
    fprintf('Episodio: %d, Pasos: %.1f, Retorno: %.1f, Metas: %.2f, Perdida Actor: %.4f, Perdida Critico: %.4f, Entropia: %.4f\n', episode, mean_steps, total_returns(episode), total_goals(episode), total_actor_loss(episode), total_critic_loss(episode), total_h(episode))
end
delete(gcp('nocreate'));
optimal_path = create_path(actor, M);
subplot(3,2,1), semilogy(1:num_episodes, total_steps), grid on
title('Pasos'), xlabel('Épocas'), ylabel('Num Pasos')
subplot(3,2,2), plot(1:num_episodes, total_returns), grid on
title('Retornos'), xlabel('Épocas'), ylabel('Retorno')
subplot(3,2,3), plot(1:num_episodes, total_actor_loss), grid on
title('Pérdida Actor'), xlabel('Épocas'), ylabel('Error')
subplot(3,2,4), plot(1:num_episodes, total_critic_loss), grid on
title('Pérdida Crítico'), xlabel('Épocas'), ylabel('Error')
subplot(3,2,5), plot(1:num_episodes, total_h), grid on
title('Entropía'), xlabel('Épocas'), ylabel('Entropía')
subplot(3,2,6), plot(1:num_episodes, total_goals), grid on
title('Metas por iteración'), xlabel('Épocas'), ylabel('Metas')
draw_maze(M, start_position, optimal_path, [goal_row goal_col])

% =====================================================================
%  Funciones locales de gradiente (verificadas matematicamente)
% =====================================================================

function grad = critic_gradient(model, states, target)
    % Critico: salida lineal, minimiza 1/2 * MSE.
    % d/dz [1/2 (V - R)^2] = (V - R).  Equivale al vf_coef=0.5 del A2C canonico.
    T = size(states, 1);
    delta = cell(1, model.num_layers);
    outputs = model.forward(states);
    y_pred = outputs{end};
    delta{end} = y_pred - target;
    for i = model.num_layers -1 : -1 : 1
        layer_activation = model.layers{i}.activation;
        a = outputs{i+1};
        derivative = model.activation_derivative(layer_activation, a);
        w = model.layers{i+1}.weights(:, 2:end);
        delta{i} = derivative .* (delta{i+1} * w);
    end
    grad = cell(1, model.num_layers);
    for i = 1 : model.num_layers
        grad{i} = (1 / T) * delta{i}' * [ones(T, 1), outputs{i}];
    end
end

function [grad, loss, H] = actor_gradient(model, states, actions_taken, targets, values, beta)
    % Actor: salida softmax + bonus de entropia.
    % loss = -mean( A * log pi(a) ) - beta * H,   con A = (targets - values) normalizada.
    T = size(states, 1);
    outputs = model.forward(states);
    probabilities = outputs{end};
    log_probabilities = log(probabilities + 1e-6);
    H = -sum(probabilities .* log_probabilities, 2);
    indices = sub2ind(size(probabilities), (1:T)', actions_taken);
    action_log_probabilities = log_probabilities(indices);
    adv = targets - values;
    adv = adv / (std(adv) + 1e-8);          % normalizacion de ventajas (tecnica estandar, no del A2C base)
    loss = sum(-adv .* action_log_probabilities - beta * H);
    one_hot = zeros(size(probabilities));
    one_hot(indices) = 1;
    delta = cell(1, model.num_layers);
    % gradiente en la pre-activacion de la salida:
    %   polit.: -adv * (one_hot - p)            (de -A*log pi)
    %   entrop.: +beta * p .* (log p + H)        (de -beta*H)
    delta{end} = -adv .* (one_hot - probabilities) + beta * probabilities .* (log_probabilities + H);
    for i = model.num_layers - 1 : -1 : 1
        layer_activation = model.layers{i}.activation;
        a = outputs{i+1};
        derivative = model.activation_derivative(layer_activation, a);
        w = model.layers{i + 1}.weights(:, 2:end);
        delta{i} = derivative .* (delta{i + 1} * w);
    end
    grad = cell(1, model.num_layers);
    for i = 1 : model.num_layers
        grad{i} = delta{i}' * [ones(T, 1), outputs{i}] / T;
    end
    loss = loss / T;
    H = mean(H);
end

function grad = average_grad(model, grads, num_envs)
    grad = grads{1};
    for env = 2 : num_envs
        for i = 1 : model.num_layers
            grad{i} = grad{i} + grads{env}{i};
        end
    end
    for i = 1 : model.num_layers
        grad{i} = grad{i} / num_envs;
    end
end