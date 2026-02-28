close all; clear; clc

% Carga del laberinto y definición de acciones: arriba, derecha, abajo, izquierda
M = create_maze();
actions = [-1 0; 0 1; 1 0; 0 -1];

start_position = [1 2];
[goal_row, goal_col] = find(M==10);

[m, n] = size(M);
num_actions = length(actions);

% Parámetros del algoritmo
alpha = 0.1;        % Tasa de aprendizaje
gamma = 0.99;       % Factor de descuento
epsilon = 1;        % Parámetro de exploración inicial
decay = 0.99;       % Decaimiento de epsilon
num_episodes = 1000;
N = 15;             % Número de pasos para n-step SARSA
max_steps = 10000;  % Límite de pasos por episodio (evita loops infinitos)

% Inicialización de la tabla Q
Q = zeros(m, n, num_actions);

% Algoritmo n-step SARSA (On-policy TD Control con n pasos)
% Actualización: Q(S_tau, A_tau) = Q(S_tau, A_tau) + alpha * [G - Q(S_tau, A_tau)]
% donde G = sum_{i=1}^{min(n, T-tau)} gamma^{i-1} * R_{tau+i} + gamma^n * Q(S_{tau+n}, A_{tau+n}) (si tau+n < T)
for episode = 1 : num_episodes
    epsilon = max(0.1, decay * epsilon);
    state = start_position;
    
    % Seleccionar acción inicial A_0
    action = egreedy_action(epsilon, Q, state, num_actions);
    
    % Almacenar trayectoria del episodio
    % states(1) = S_0, states(k) = S_{k-1}
    % actions_taken(1) = A_0, actions_taken(k) = A_{k-1}
    % rewards(1) = R_1, rewards(k) = R_k (recompensa después de A_{k-1})
    states = [];
    actions_taken = [];
    rewards = [];
    
    states(end + 1, :) = state;
    actions_taken(end + 1) = action;
    
    T = inf;            % Tiempo de terminación (infinito mientras no termine)
    t = 0;              % Contador de tiempo (paso actual)
    step_count = 0;     % Contador de pasos (protección anti-loop)
    
    while step_count < max_steps
        if t < T
            % Tomar acción A_t, observar R_{t+1}, S_{t+1}
            [next_state, reward, done] = step(M, state, action, actions, m, n);
            states(end + 1, :) = next_state;      % Guardar S_{t+1}
            rewards(end + 1) = reward;            % Guardar R_{t+1}
            step_count = step_count + 1;
            
            if done || isequal(next_state, [goal_row goal_col])
                % Estado terminal: episodio termina en tiempo T = t+1
                T = t + 1;
            else
                % Seleccionar siguiente acción A_{t+1} usando política epsilon-greedy
                next_action = egreedy_action(epsilon, Q, next_state, num_actions);
                actions_taken(end + 1) = next_action;  % Guardar A_{t+1}
            end
            
            % Avanzar al siguiente estado
            state = next_state;
            if ~done && T == inf
                action = next_action;  % Solo actualizar acción si no es terminal
            end
        end
        
        % Calcular tau: el tiempo cuyo par (S_tau, A_tau) se actualiza
        % tau = t - N + 1
        tau = t - N + 1;
        
        if tau >= 0
            % Calcular retorno N-step G_{tau:tau+N}
            % Sumar recompensas desde R_{tau+1} hasta R_{min(tau+N, T)}
            G = 0;
            upper_bound = min(tau + N, T);  % Incluir recompensa en T si existe
            for i = (tau + 1) : upper_bound
                G = G + gamma^(i - tau - 1) * rewards(i);
            end
            
            % Si no hemos llegado al terminal, agregar valor bootstrap
            % Q(S_{tau+N}, A_{tau+N}) está en índice tau+N+1 (porque MATLAB es 1-indexed)
            if tau + N < T
                s_idx = tau + N + 1;  % Índice de S_{tau+N} en states
                a_idx = tau + N + 1;  % Índice de A_{tau+N} en actions_taken
                G = G + gamma^N * Q(states(s_idx, 1), states(s_idx, 2), actions_taken(a_idx));
            end
            
            % Actualizar Q(S_tau, A_tau)
            % states(tau+1) = S_tau, actions_taken(tau+1) = A_tau
            s_update = tau + 1;
            Q(states(s_update, 1), states(s_update, 2), actions_taken(s_update)) = ...
                Q(states(s_update, 1), states(s_update, 2), actions_taken(s_update)) + ...
                alpha * (G - Q(states(s_update, 1), states(s_update, 2), actions_taken(s_update)));
        end
        
        t = t + 1;
        
        % Terminar cuando hayamos actualizado el último paso (tau = T-1)
        if tau == T - 1
            break;
        end
    end
    
    fprintf('Episodio: %d\n', episode)
end

% Extraer política greedy final
[~, policy] = max(Q, [], 3);
policy(M==-2) = 0;
policy(M==10) = 0;

plot_q_values(Q)

draw_maze(M, start_position, policy, [goal_row goal_col])
