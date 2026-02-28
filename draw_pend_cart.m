function draw_pend_cart(currentState, cartMass_val, pendulumMass_val, rodLength)
% draw_pendulum_on_cart_v2: Visualiza un sistema de carro-péndulo.
%
% Entradas:
%   currentState: Un vector que contiene el estado del sistema.
%                 currentState(1) = posición x del carro
%                 currentState(3) = ángulo del péndulo (theta) respecto a la vertical
%   cartMass_val: Masa del carro.
%   pendulumMass_val: Masa de la lenteja del péndulo.
%   rodLength: Longitud de la barra del péndulo.

% --- Extracción de las variables de estado ---
cart_x_pos = currentState(1);   % Posición horizontal del centro del carro
pendulum_theta = currentState(3); % Ángulo del péndulo (radianes), 0 es vertical hacia abajo

% --- Definición de Dimensiones Basadas en las Masas (con algunos ajustes) ---
% Dimensiones del Carro
cartWidth = 1.5 * sqrt(cartMass_val / 5);  % Ancho del carro (ajustado)
cartHeight = 0.5 * sqrt(cartMass_val / 5); % Altura del carro (ajustado)

% Dimensiones de las Ruedas
wheelRadius = 0.18; % Radio de las ruedas (valor modificado)
wheelDiameter = 2 * wheelRadius;

% Dimensiones de la Lenteja del Péndulo
pendulumBobDiameter = 0.38 * sqrt(pendulumMass_val); % Diámetro de la lenteja (ajustado)

% --- Cálculo de Posiciones Clave ---
% Posición vertical del centro del carro
% (las ruedas están sobre el suelo en y=0)
cart_y_center = wheelRadius + cartHeight / 2;

% Coordenadas del extremo de la barra del péndulo (centro de la lenteja)
% El péndulo pivota desde el centro del carro (cart_x_pos, cart_y_center)
pendulum_pivot_x = cart_x_pos;
pendulum_pivot_y = wheelRadius + cartHeight;

bob_center_x = pendulum_pivot_x + rodLength * sin(pendulum_theta);
bob_center_y = pendulum_pivot_y - rodLength * cos(pendulum_theta); % Negativo porque el coseno es desde la vertical hacia abajo

% --- Inicio del Dibujo ---
% Limpiar la figura actual para la nueva animación (si se usa en un bucle)

% Dibujar el Suelo (línea de referencia)
plot([-15 15], [0 0], 'Color', [0.4 0.4 0.4], 'LineStyle', '-.', 'LineWidth', 1); % Estilo modificado
hold on; % Mantener los elementos gráficos para dibujar múltiples componentes
grid on

% Dibujar el Cuerpo del Carro
rectangle('Position', [cart_x_pos - cartWidth / 2, cart_y_center - cartHeight / 2, cartWidth, cartHeight], ...
          'Curvature', 0.15, ... % Curvatura de las esquinas ligeramente aumentada
          'FaceColor', [0.3 0.7 0.9], ... % Nuevo color azulado
          'EdgeColor', [0.1 0.3 0.5], ... % Color de borde más oscuro
          'LineWidth', 1.8);      % Grosor de línea aumentado

% Dibujar las Ruedas
% Las ruedas se dibujan con su base en y=0
wheel_offset_factor = 0.70; % Factor para la separación de las ruedas del centro del carro

% Rueda Izquierda
left_wheel_center_x = cart_x_pos - cartWidth / 2 * wheel_offset_factor;
rectangle('Position', [left_wheel_center_x - wheelRadius, 0, wheelDiameter, wheelDiameter], ...
          'Curvature', 1, ...    % Círculo perfecto
          'FaceColor', [0.2 0.2 0.2], ... % Color de rueda más oscuro
          'EdgeColor', [0.05 0.05 0.05], ...
          'LineWidth', 1.2);

% Rueda Derecha
right_wheel_center_x = cart_x_pos + cartWidth / 2 * wheel_offset_factor;
rectangle('Position', [right_wheel_center_x - wheelRadius, 0, wheelDiameter, wheelDiameter], ...
          'Curvature', 1, ...
          'FaceColor', [0.2 0.2 0.2], ...
          'EdgeColor', [0.05 0.05 0.05], ...
          'LineWidth', 1.2);

% Dibujar la Barra del Péndulo
plot([pendulum_pivot_x, bob_center_x], [pendulum_pivot_y, bob_center_y], ...
     'Color', [0.7 0.2 0.2], 'LineWidth', 3); % Color y grosor modificados

% Dibujar la Lenteja del Péndulo (Bob)
rectangle('Position', [bob_center_x - pendulumBobDiameter / 2, bob_center_y - pendulumBobDiameter / 2, pendulumBobDiameter, pendulumBobDiameter], ...
          'Curvature', 1, ...    % Círculo perfecto
          'FaceColor', [0.9 0.4 0.1], ... % Nuevo color anaranjado
          'EdgeColor', [0.6 0.2 0.0], ...
          'LineWidth', 1.8);

% --- Ajustes Finales del Gráfico ---
axis([-5 5 -2 3])
axis equal
set(gcf,'Position',[100 100 1000 400])
drawnow
hold off

end