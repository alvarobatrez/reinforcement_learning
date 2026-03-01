%% TEST_PEND_CART Script de prueba para el modelo de péndulo invertido
%
%   Este script simula la dinámica de un péndulo invertido sobre carro
%   sin control (u = 0), mostrando cómo el péndulo cae desde una
%   condición inicial cercana a la posición invertida.
%
%   Para estabilizar el péndulo, se necesitaría implementar un controlador
%   (por ejemplo, LQR o PID) que calcule u en función del estado.

close all; clear; clc;

%% --- PARÁMETROS DEL SISTEMA ---
% Masas
M = 5;      % Masa del carro [kg]
m = 1;      % Masa del péndulo (concentrada en la punta) [kg]

% Geometría
L = 2;      % Longitud de la varilla del péndulo [m]
g = 9.8;    % Aceleración de la gravedad [m/s²]

% Fricción
b = 0.01;   % Coeficiente de fricción del carro con el riel [N·s/m]
d = 0.5;    % Coeficiente de fricción en la articulación [N·m·s/rad]

%% --- CONFIGURACIÓN DE LA SIMULACIÓN ---
tspan = 0:0.02:10;      % Vector de tiempo: de 0 a 10 s con paso de 0.02 s
                        % El paso pequeño asegura buena precisión en ode45

% Condición inicial: [posición; velocidad; ángulo; velocidad_angular]
% theta = pi corresponde al péndulo perfectamente invertido (hacia arriba)
% theta = pi + 0.5 = 3.64 rad ≈ 208.6° (ligeramente desplazado de vertical)
x0 = [0;      % x0 = 0 m (carro en el origen)
      0;      % v0 = 0 m/s (carro en reposo)
      pi+0.5; % theta0 ≈ 208.6° (péndulo ligeramente inclinado desde arriba)
      0];     % omega0 = 0 rad/s (péndulo inicialmente sin velocidad angular)

%% --- ENTRADA DE CONTROL ---
% u = 0 significa que no hay fuerza aplicada al carro (sistema libre)
% El péndulo caerá por gravedad y oscilará hasta que la fricción lo detenga
u = 0;      % Fuerza de control [N]

%% --- SIMULACIÓN NUMÉRICA ---
% Crear función anónima para pasar parámetros adicionales a ode45
% Sintaxis: @(t,x) función(t, x, parámetros_adicionales...)
fun = @(t,x) pend_cart(t, x, M, m, L, g, b, d, u);

% Integración numérica usando Runge-Kutta de orden 4-5
[t, x] = ode45(fun, tspan, x0);
%   t  = vector de tiempos donde se calculó la solución
%   x  = matriz de estados, cada fila es el estado en el tiempo t(i)
%        columnas: [posición, velocidad, ángulo, velocidad_angular]

%% --- RESULTADOS EN CONSOLA ---
fprintf('=== Simulación completada ===\n');
fprintf('Duración total: %.1f segundos\n', t(end));
fprintf('Estado final:\n');
fprintf('  Posición del carro:     %.3f m\n', x(end, 1));
fprintf('  Velocidad del carro:    %.3f m/s\n', x(end, 2));
fprintf('  Ángulo del péndulo:     %.3f rad (%.1f°)\n', x(end, 3), rad2deg(x(end, 3)));
fprintf('  Velocidad angular:      %.3f rad/s\n', x(end, 4));
fprintf('\nNota: Sin control (u=0), el péndulo oscila hasta que la fricción lo detiene.\n');

%% --- VISUALIZACIÓN/ANIMACIÓN ---
% Nota: La animación puede hacerse más rápida saltando frames
frame_skip = 1;  % Dibujar cada n-ésimo frame (aumentar para acelerar)

fprintf('\nIniciando animación...\n');
for i = 1:frame_skip:length(t)
    draw_pend_cart(x(i,:), M, m, L);
    
    % Opcional: pausa para mantener tiempo real con la simulación
    % pause(0.02);  % Descomentar para sincronización aproximada
end
fprintf('Animación completada.\n');

%% --- GRÁFICAS DE ESTADO (opcional, para análisis) ---
figure('Name', 'Estados del sistema', 'Position', [100 550 1200 400]);

% Posición del carro
subplot(2, 2, 1);
plot(t, x(:,1), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Tiempo (s)');
ylabel('Posición (m)');
title('Posición del carro');

% Velocidad del carro
subplot(2, 2, 2);
plot(t, x(:,2), 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Tiempo (s)');
ylabel('Velocidad (m/s)');
title('Velocidad del carro');

% Ángulo del péndulo
subplot(2, 2, 3);
plot(t, rad2deg(x(:,3)), 'g-', 'LineWidth', 1.5);
grid on;
xlabel('Tiempo (s)');
ylabel('Ángulo (°)');
title('Ángulo del péndulo (0°=abajo, 180°=arriba)');

% Velocidad angular del péndulo
subplot(2, 2, 4);
plot(t, x(:,4), 'm-', 'LineWidth', 1.5);
grid on;
xlabel('Tiempo (s)');
ylabel('Velocidad angular (rad/s)');
title('Velocidad angular del péndulo');
