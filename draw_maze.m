function draw_maze(maze, agent_position, policy, exit)

actions = [-1 0; 0 1; 1 0; 0 -1];

maze(maze == -1) = 1;
maze(maze == -2) = 2;
maze(exit(1), exit(2)) = 3;

figure

colormap([1 1 1;
          0 0 0;
          1 0 0]);

imagesc(maze)
hold on
axis off
axis equal

agent_marker = plot(agent_position(2), agent_position(1), 'bo', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
title('Laberinto')

max_steps = 100;
for step = 1 : max_steps
    pause(0.25)

    if isequal(agent_position, exit)
        break
    end

    policy_selected = policy(agent_position(1), agent_position(2));
    agent_position = agent_position + actions(policy_selected, :);
    
    set(agent_marker, 'XData', agent_position(2), 'YData', agent_position(1));
    drawnow
end