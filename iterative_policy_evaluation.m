close all; clear, clc

R = [0 -1 -1 -1; -1 -1 -1 -1; -1 -1 -1 -1; -1 -1 -1 0];
actions = [-1 0; 0 1; 1 0; 0 -1];

[m, n] = size(R);
num_actions = length(actions);

prob = 0.25;
theta = 0.001;
gamma = 0.9;

V = zeros(m, n);
Vp = zeros(m, n);

while true
    delta = 0;

    for i = 1 : m
        for j = 1 : n

            if R(i, j) == 0
                continue
            end

            v = V(i,j);
            suma = 0;
    
            for action = 1 : num_actions
    
                new_i = i + actions(action, 1);
                new_j = j + actions(action, 2);
            
                if new_i < 1 || new_i > m || new_j < 1 || new_j > n
                    new_i = i;
                    new_j = j;
                end
        
                suma = suma + prob * (R(new_i, new_j) + gamma * Vp(new_i, new_j));
            end
            
            V(i,j) = suma;
            delta = max(delta, abs(v - V(i, j)));
        end
    end

    Vp = V;

    if delta < theta
        break;
    end
end

disp('Matriz de Recompensas')
disp(R)

disp('Función de Valor')
disp(V)