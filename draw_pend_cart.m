function draw_pend_cart(currentState, cartMass_val, pendulumMass_val, rodLength)
% DRAW_PEND_CART Visualiza el sistema de péndulo invertido sobre carro.
%
%   Esta función dibuja una animación 2D del sistema carro-péndulo con
%   dimensiones proporcionales a las masas del sistema.
%
%   Entradas:
%       currentState      - Vector de estado [x; v; theta; omega]
%                           currentState(1) = posición x del centro del carro (m)
%                           currentState(2) = velocidad del carro (m/s) - no usado en visualización
%                           currentState(3) = ángulo theta del péndulo (rad)
%                                             0 = colgando hacia abajo, pi = invertido hacia arriba
%                           currentState(4) = velocidad angular (rad/s) - no usado en visualización
%       cartMass_val      - Masa del carro (kg), determina tamaño visual del carro
%       pendulumMass_val  - Masa de la lenteja del péndulo (kg), determina su diámetro
%       rodLength         - Longitud de la varilla del péndulo (m)
%
%   Nota: La función usa hold on/off y limpia la figura en cada llamada.
%         Diseñada para usarse dentro de un bucle de animación.

    %% --- EXTRACCIÓN DE VARIABLES DE ESTADO ---
    cart_x_pos = currentState(1);       % Posición horizontal del centro del carro [m]
    pendulum_theta = currentState(3);   % Ángulo del péndulo [rad]
    
    %% --- DEFINICIÓN DE DIMENSIONES VISUALES ---
    % Las dimensiones se escalan con la raíz cuadrada de la masa para que
    % el área (aproximadamente) sea proporcional a la masa.
    
    % Dimensiones del carro
    cartWidth = 1.5 * sqrt(cartMass_val / 5);   % Ancho proporcional a la masa [m]
    cartHeight = 0.5 * sqrt(cartMass_val / 5);  % Altura proporcional a la masa [m]
    
    % Dimensiones de las ruedas
    wheelRadius = 0.18;     % Radio fijo para mejor estética [m]
    wheelDiameter = 2 * wheelRadius;
    
    % Dimensiones de la lenteja del péndulo (masa puntual)
    pendulumBobDiameter = 0.38 * sqrt(pendulumMass_val);  % Diámetro proporcional a sqrt(masa) [m]
    
    %% --- CÁLCULO DE POSICIONES GEOMÉTRICAS ---
    % El suelo está en y = 0
    cart_y_center = wheelRadius + cartHeight / 2;  % Centro vertical del carro [m]
    
    % Posición del pivote (punto de unión carro-péndulo)
    pendulum_pivot_x = cart_x_pos;
    pendulum_pivot_y = wheelRadius + cartHeight;   % En la parte superior del carro [m]
    
    % Posición de la lenteja del péndulo (extremo de la varilla)
    % Usamos convención: theta = 0 -> colgando abajo, theta = pi -> arriba
    % x: seno para desplazamiento horizontal
    % y: -coseno para que 0 sea abajo y pi sea arriba
    bob_center_x = pendulum_pivot_x + rodLength * sin(pendulum_theta);
    bob_center_y = pendulum_pivot_y - rodLength * cos(pendulum_theta);
    
    %% --- DIBUJO DE LOS ELEMENTOS ---
    % Limpiar figura anterior y configurar ejes
    clf;  % Clear current figure para animación fluida
    hold on;
    grid on;
    
    % Suelo (línea de referencia)
    plot([-15 15], [0 0], 'Color', [0.4 0.4 0.4], 'LineStyle', '-.', 'LineWidth', 1);
    
    % Carro (rectángulo con esquinas redondeadas)
    rectangle('Position', [cart_x_pos - cartWidth/2, cart_y_center - cartHeight/2, cartWidth, cartHeight], ...
              'Curvature', 0.15, ...
              'FaceColor', [0.3 0.7 0.9], ...    % Azul claro
              'EdgeColor', [0.1 0.3 0.5], ...    % Borde azul oscuro
              'LineWidth', 1.8);
    
    % Ruedas (círculos en la base del carro)
    wheel_offset_factor = 0.70;  % Posición de las ruedas como fracción del ancho
    
    % Rueda izquierda
    left_wheel_center_x = cart_x_pos - cartWidth/2 * wheel_offset_factor;
    rectangle('Position', [left_wheel_center_x - wheelRadius, 0, wheelDiameter, wheelDiameter], ...
              'Curvature', 1, ...                % Círculo perfecto
              'FaceColor', [0.2 0.2 0.2], ...    % Gris oscuro
              'EdgeColor', [0.05 0.05 0.05], ...
              'LineWidth', 1.2);
    
    % Rueda derecha
    right_wheel_center_x = cart_x_pos + cartWidth/2 * wheel_offset_factor;
    rectangle('Position', [right_wheel_center_x - wheelRadius, 0, wheelDiameter, wheelDiameter], ...
              'Curvature', 1, ...
              'FaceColor', [0.2 0.2 0.2], ...
              'EdgeColor', [0.05 0.05 0.05], ...
              'LineWidth', 1.2);
    
    % Varilla del péndulo (línea roja gruesa)
    plot([pendulum_pivot_x, bob_center_x], [pendulum_pivot_y, bob_center_y], ...
         'Color', [0.7 0.2 0.2], 'LineWidth', 3);
    
    % Lenteja del péndulo (círculo naranja)
    rectangle('Position', [bob_center_x - pendulumBobDiameter/2, bob_center_y - pendulumBobDiameter/2, ...
                           pendulumBobDiameter, pendulumBobDiameter], ...
              'Curvature', 1, ...                % Círculo perfecto
              'FaceColor', [0.9 0.4 0.1], ...    % Naranja
              'EdgeColor', [0.6 0.2 0.0], ...
              'LineWidth', 1.8);
    
    %% --- CONFIGURACIÓN FINAL DE LA FIGURA ---
    axis([-5 5 -2 3]);      % Límites de visualización [m]
    axis equal;             % Escalas iguales en x e y
    xlabel('Posición x (m)');
    ylabel('Altura y (m)');
    title(sprintf('Péndulo Invertido - t = %.2f s, θ = %.2f°', ...
                  currentState(1), rad2deg(pendulum_theta)));
    set(gcf, 'Position', [100 100 1000 400]);  % Tamaño de ventana [px]
    drawnow;                % Forzar actualización inmediata
    hold off;
end
