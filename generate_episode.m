function [states, actions_taken, rewards] = generate_episode(M, pi, start_position, goal_position, actions, num_actions, max_steps, m, n)
    % Genera un episodio siguiendo la política pi desde start_position hasta goal_position
    % o hasta alcanzar max_steps pasos
    
    state = start_position;
    i = state(1);
    j = state(2);
    states = zeros(max_steps, 2);
    actions_taken = zeros(max_steps, 1);
    rewards = zeros(max_steps, 1);
    step = 1;
    
    while ~isequal(state, goal_position) && step <= max_steps
        states(step, :) = state;
        
        % Muestrear acción según la política ε-soft
        actions_probabilities = squeeze(pi(i, j, :));
        action = randsample(1:num_actions, 1, true, actions_probabilities);
        
        % Calcular siguiente estado tentativo
        next_i = i + actions(action, 1);
        next_j = j + actions(action, 2);
        
        % Manejo de límites y paredes: rebote con penalización
        if next_i < 1 || next_i > m || next_j < 1 || next_j > n || M(next_i, next_j) == -2
            reward = -2;        % Penalización por chocar
            % Permanece en el mismo estado (rebote)
        else
            reward = M(next_i, next_j);  % Recompensa del estado destino
            i = next_i;         % Actualiza posición
            j = next_j;
        end
        
        state = [i j];
        actions_taken(step) = action;
        rewards(step) = reward;
        
        step = step + 1;
    end
    
    % Truncar vectores al tamaño real del episodio
    states = states(1 : step-1, :);
    actions_taken = actions_taken(1 : step-1);
    rewards = rewards(1 : step-1);
end
