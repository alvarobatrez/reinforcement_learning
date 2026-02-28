close all; clear; clc

% Parámetros del motor DC
J = 3.4e-5;     % Momento de inercia del rotor [kg*m^2]
b = 2.2e-5;     % Coeficiente de fricción viscosa [N*m*s]
K = 50e-3;      % Constante de torque y back-EMF [V/(rad/s) o N*m/A]
L = 7.7e-3;     % Inductancia [H]
R = 11.4;       % Resistencia [Ohm]

% Parámetros del espacio de estados (voltaje, velocidad angular)
v_min = 0; v_max = 1;       % Rango de voltaje de entrada [V]
w_min = 0; w_max = 20;      % Rango de velocidad angular [rad/s]
bins_v = 50;                % Número de bins para voltaje
bins_w = 50;                % Número de bins para velocidad angular
low = [v_min w_min];
high = [v_max w_max];
n_tilings = 4;              % Número de tilings (códigos de mosaico)

% Crear tilings con offsets aleatorios para aproximación de función
[tilings_v, tilings_w] = create_tilings([bins_v bins_w], low, high, n_tilings);

% Acciones: cambios discretos en voltaje
actions = [-0.01 0 0.01];   % Reducir, mantener, aumentar voltaje
num_actions = length(actions);

% Parámetros de control
ref = 10;                   % Velocidad de referencia objetivo [rad/s]
Ts = 0.01;                  % Tiempo de muestreo [s]
T = 1;                      % Duración del episodio [s]
t_steps = T / Ts;           % Número de pasos por episodio

% Parámetros de SARSA
alpha = 0.1 / n_tilings;    % Tasa de aprendizaje ajustada por número de tilings
gamma = 0.99;               % Factor de descuento
epsilon = 1;                % Parámetro de exploración inicial
decay = 0.99;               % Decaimiento de epsilon
num_episodes = 2500;

% Inicialización de Q: Q(tiling, v_bin, w_bin, action)
% Usamos tile coding: cada estado activa exactamente un tile por tiling
Q = zeros(n_tilings, bins_v, bins_w, num_actions);

sum_returns = zeros(num_episodes, 1);

% Algoritmo SARSA con Tile Coding (aproximación de función lineal)
% Aproximación: Q(s,a) ≈ sum_i Q_i(s_i, a) donde s_i es el tile activo en el tiling i
for episode = 1 : num_episodes
    epsilon = max(0.01, decay * epsilon);
    
    % Condiciones iniciales
    v = 0;                      % Voltaje aplicado [V]
    w = 0;                      % Velocidad angular [rad/s]
    i = 0;                      % Corriente [A]
    state = discretize_state([v w], tilings_v, tilings_w);
    
    G = 0;                      % Retorno acumulado del episodio
    
    % Seleccionar acción inicial usando política epsilon-greedy
    action = egreedy_action(epsilon, Q, state, num_actions, n_tilings);
    
    for t = 1 : t_steps
        % Aplicar acción: modificar voltaje y limitar al rango permitido
        v = clamp_voltage(v + actions(action), v_min, v_max);
        
        % Simular motor durante Ts segundos
        [w, i] = simulate_motor([w i], Ts, J, b, K, L, R, v);
        
        % Discretizar nuevo estado continuo a índices de tiles
        next_state = discretize_state([v w], tilings_v, tilings_w);
        
        % Seleccionar siguiente acción (on-policy)
        next_action = egreedy_action(epsilon, Q, next_state, num_actions, n_tilings);
        
        % Calcular recompensa: negativo del error cuadrático
        % Maximizar recompensa = minimizar (ref - w)^2
        reward = -(ref - w)^2;
        
        % SARSA con Tile Coding: Calcular error TD usando valores totales
        % Q_total(s,a) = suma de Q_i(s_i, a) sobre todos los tilings i
        current_Q = 0;
        next_Q = 0;
        for n = 1 : n_tilings
            current_Q = current_Q + Q(n, state(n, 1), state(n, 2), action);
            next_Q = next_Q + Q(n, next_state(n, 1), next_state(n, 2), next_action);
        end
        
        % Error TD: delta = R + gamma * Q(s',a') - Q(s,a)
        delta = reward + gamma * next_Q - current_Q;
        
        % Actualizar todos los tilings activos con el mismo delta
        % Cada tiling contribuye con gradiente 1 a la aproximación lineal
        for n = 1 : n_tilings
            Q(n, state(n, 1), state(n, 2), action) = Q(n, state(n, 1), state(n, 2), action) + ...
                alpha * delta;
        end
        
        state = next_state;
        action = next_action;
        
        G = G + reward;
    end
    
    sum_returns(episode) = G;
    
    fprintf('Episodio: %d\n', episode)
end

% Visualizar respuesta con la política aprendida
initial_conditions = [0 0 0]; % [v, w, i]
draw_response(Q, actions, num_actions, initial_conditions, n_tilings, tilings_v, tilings_w, v_min, v_max, t_steps, Ts, J, b, K, L, R)

% Gráfica de convergencia
figure, plot(1:num_episodes, sum_returns)
xlabel('Episodio'), ylabel('Retornos'), title('SARSA con Tile Coding - Control Motor DC')

t = (0 : t_steps-1) * Ts;

function [tilings_v, tilings_w] = create_tilings(bins, low, high, n)
    % Crea n tilings (rejillas desplazadas) para tile coding
    % Cada tiling tiene un offset aleatorio para generalizar mejor
    
    tilings_v = cell(n, 1);
    tilings_w = cell(n, 1);
    
    for i = 1 : n
        % Offset aleatorio del 20% para cada tiling
        low_i = low - rand * 0.2 * low;
        high_i = high + rand * 0.2 * high;
        
        % Crear centros y bordes de los bins
        v_centers = linspace(low_i(1), high_i(1), bins(1));
        w_centers = linspace(low_i(2), high_i(2), bins(2));
        
        % Bordes son los puntos medios entre centros
        v_edges = zeros(1, bins(1) + 1);
        v_edges(1) = low_i(1);
        v_edges(end) = high_i(1);
        for j = 2 : bins(1)
            v_edges(j) = (v_centers(j-1) + v_centers(j)) / 2;
        end
        
        w_edges = zeros(1, bins(2) + 1);
        w_edges(1) = low_i(2);
        w_edges(end) = high_i(2);
        for j = 2 : bins(2)
            w_edges(j) = (w_centers(j-1) + w_centers(j)) / 2;
        end
        
        tilings_v{i} = v_edges;
        tilings_w{i} = w_edges;
    end
end

function state = discretize_state(observations, tilings_v, tilings_w)
    % Discretiza un estado continuo (v, w) a índices de tiles
    % Para cada tiling, retorna el índice del bin que contiene el valor
    
    v = observations(1);
    w = observations(2);
    n_tilings = length(tilings_v);
    state = zeros(n_tilings, 2);
    
    for i = 1 : n_tilings
        % Encontrar índice para voltaje
        v_edges = tilings_v{i};
        v_idx = find(v >= v_edges(1:end-1) & v < v_edges(2:end), 1, 'first');
        if isempty(v_idx)
            v_idx = length(v_edges) - 1;  % Caso límite: último bin
        end
        
        % Encontrar índice para velocidad angular
        w_edges = tilings_w{i};
        w_idx = find(w >= w_edges(1:end-1) & w < w_edges(2:end), 1, 'first');
        if isempty(w_idx)
            w_idx = length(w_edges) - 1;  % Caso límite: último bin
        end
        
        state(i, :) = [v_idx, w_idx];
    end
end

function action = egreedy_action(epsilon, Q, state, num_actions, n)
    % Selecciona acción usando política epsilon-greedy
    % El valor Q(s,a) es la suma de los valores de todos los tilings activos
    
    if rand > epsilon
        % Exploit: sumar contribuciones de todos los tilings
        total_action_values = zeros(1, num_actions);
        for i = 1 : n
            av = Q(i, state(i,1), state(i,2), :);
            av = reshape(av, [1, num_actions]);
            total_action_values = total_action_values + av;
        end
        [~, action] = max(total_action_values);
    else
        % Explore: acción aleatoria
        action = randi(num_actions);
    end
end

function v = clamp_voltage(v, v_min, v_max)
    % Limita el voltaje al rango [v_min, v_max]
    v = max(v_min, min(v, v_max));
end

function draw_response(Q, actions, num_actions, initial_conditions, n_tilings, tilings_v, tilings_w, v_min, v_max, t_steps, Ts, J, b, K, L, R)
    % Simula y grafica la respuesta del sistema con la política greedy aprendida
    
    v = initial_conditions(1);
    w = initial_conditions(2);
    i = initial_conditions(3);
    state = discretize_state([v w], tilings_v, tilings_w);
    vel_motor = zeros(t_steps, 1);
    
    for t = 1 : t_steps
        % Calcular valores Q para todas las acciones (sumando tilings)
        q = zeros(num_actions, 1);
        for a = 1 : num_actions
            for j = 1 : n_tilings
                q(a) = q(a) + Q(j, state(j,1), state(j,2), a);
            end
        end
        
        % Seleccionar acción greedy
        [~, action] = max(q);
        v = clamp_voltage(v + actions(action), v_min, v_max);
        [w, i] = simulate_motor([w i], Ts, J, b, K, L, R, v);
        state = discretize_state([v w], tilings_v, tilings_w);
        
        vel_motor(t) = w;
    end
    
    t = (0 : t_steps-1) * Ts;
    figure, plot(t, vel_motor), grid on
    xlabel('Tiempo [s]'), ylabel('Velocidad angular [rad/s]'), title('Respuesta del Motor DC con Política Aprendida')
end
