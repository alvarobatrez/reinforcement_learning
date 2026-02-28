function draw_maze(maze, agent_position, policy, exit)

actions = [-1 0; 0 1; 1 0; 0 -1];

% Preparar matriz para visualización
maze(maze == -1) = 1;   % Caminos -> blanco
maze(maze == -2) = 2;   % Paredes -> negro
maze(exit(1), exit(2)) = 3;  % Salida -> rojo

figure

% Colormap: [blanco; negro; rojo]
colormap([1 1 1;
          0 0 0;
          1 0 0]);

imagesc(maze)
hold on
axis off
axis equal

% Dibujar agente
agent_marker = plot(agent_position(2), agent_position(1), 'bo', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
title('Laberinto')

[m, n] = size(maze);
max_steps = 100;

for step = 1 : max_steps
    pause(0.25)

    % Verificar si llegó a la salida
    if isequal(agent_position, exit)
        break
    end

    % Obtener acción de la política
    policy_selected = policy(agent_position(1), agent_position(2));
    
    % Calcular nueva posición
    next_position = agent_position + actions(policy_selected, :);
    
    % Verificar límites y paredes (safety check)
    if next_position(1) < 1 || next_position(1) > m || ...
       next_position(2) < 1 || next_position(2) > n || ...
       maze(next_position(1), next_position(2)) == 2
        % Colisión: no moverse
        next_position = agent_position;
    end
    
    agent_position = next_position;
    
    % Actualizar posición del marcador
    set(agent_marker, 'XData', agent_position(2), 'YData', agent_position(1));
    drawnow
end

end
