function dxdt = pend_cart(t, X, M, m, L, g, b, d, u)
% PEND_CART Modelo dinámico de un péndulo invertido sobre carro.
%
%   Entradas:
%       t   - Tiempo actual (no usado explícitamente, requerido por ode45)
%       X   - Vector de estado [x; v; theta; omega]
%             x     = posición del carro (m)
%             v     = velocidad del carro (m/s)
%             theta = ángulo del péndulo (rad), 0 = colgando hacia abajo, pi = invertido
%             omega = velocidad angular del péndulo (rad/s)
%       M   - Masa del carro (kg)
%       m   - Masa del péndulo (concentrada en la punta) (kg)
%       L   - Longitud de la varilla del péndulo (m)
%       g   - Aceleración de la gravedad (m/s²)
%       b   - Coeficiente de fricción viscosa del carro con el riel (N·s/m)
%       d   - Coeficiente de fricción en la articulación del péndulo (N·m·s/rad)
%       u   - Fuerza de control aplicada al carro (N)
%
%   Salida:
%       dxdt - Derivadas del estado [v; a; omega; alpha]
%
%   Ecuaciones del sistema (derivadas de la mecánica Lagrangiana):
%       (M+m)*a + m*L*alpha*cos(theta) = u - b*v + m*L*omega^2*sin(theta)
%       m*L*a*cos(theta) + m*L^2*alpha = -d*omega - m*g*L*sin(theta)
%
%   donde a = dv/dt (aceleración del carro) y alpha = domega/dt (aceleración angular)

    % Extracción de variables de estado para claridad
    v = X(2);      % Velocidad del carro
    theta = X(3);  % Ángulo del péndulo
    omega = X(4);  % Velocidad angular del péndulo
    
    % Precálculo de funciones trigonométricas para eficiencia
    s = sin(theta);
    c = cos(theta);
    
    % Matriz de masa (coeficientes de las aceleraciones)
    % Ecuación 1: (M+m)*a + (m*L*c)*alpha = términos conocidos
    % Ecuación 2: (m*L*c)*a + (m*L^2)*alpha = términos conocidos
    A = [M+m,     m*L*c;
         m*L*c,   m*L^2];
    
    % Vector del lado derecho (términos no acelerativos)
    % Fuerza neta en el carro: control - fricción + fuerza centrífuga del péndulo
    % Torque neto en el péndulo: -fricción_articulación - torque_gravedad
    B = [u - b*v + m*L*omega^2*s;
         -d*omega - m*g*L*s];
    
    % Resolución del sistema lineal A * [a; alpha] = B
    % usando división matricial izquierda para estabilidad numérica
    x = A \ B;
    
    % Construcción del vector de salida
    dxdt = [v;      % dx/dt = velocidad
            x(1);   % dv/dt = aceleración del carro
            omega;  % dtheta/dt = velocidad angular
            x(2)];  % domega/dt = aceleración angular del péndulo
end
