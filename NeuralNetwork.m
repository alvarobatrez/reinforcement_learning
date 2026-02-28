classdef NeuralNetwork
    % NeuralNetwork - Implementación de red neuronal feedforward con backpropagation
    % Soporta: SGD, Adam, AdamW
    % Funciones de activación: sigmoid, tanh, relu, linear, softmax
    % Funciones de pérdida: mse, cross_entropy
    
    properties
        layers          % Cell array con estructuras de cada capa (pesos, activación)
        learning_rate   % Tasa de aprendizaje
        optimizer       % Tipo de optimizador: 'sgd', 'adam', 'adamW'
        loss_function   % Función de pérdida: 'mse', 'cross_entropy'
        num_layers      % Número de capas (sin contar entrada)
        
        % Parámetros para Adam/AdamW
        beta1           % Decaimiento momento primero (default: 0.9)
        beta2           % Decaimiento momento segundo (default: 0.999)
        epsilon         % Constante estabilidad numérica (default: 1e-8)
        weight_decay    % Decaimiento de pesos para AdamW (default: 0.01)
        
        % Estados de optimización Adam/AdamW
        m               % Cell array: momento primero de los gradientes
        v               % Cell array: momento segundo de los gradientes  
        t = 0           % Contador de iteraciones para corrección de sesgo
    end

    methods
        function model = NeuralNetwork(input_size, layers)
            % Constructor - Inicializa la red neuronal
            % input_size: número de características de entrada
            % layers: cell array donde cada elemento es {num_neuronas, funcion_activacion}
            %         ej: {{10, 'relu'}, {1, 'sigmoid'}} para 2 capas
            
            model.num_layers = length(layers);
            model.m = cell(1, model.num_layers);
            model.v = cell(1, model.num_layers);

            for i = 1 : model.num_layers
                % Inicialización He (Kaiming): sqrt(2/n_in)
                % Adecuada para activaciones ReLU
                if i == 1
                    % Primera capa: input_size + 1 (incluye bias) entradas
                    weights = randn(layers{i}{1}, input_size + 1) * sqrt(2 / (input_size + 1));
                else
                    % Capas siguientes: neuronas_capa_anterior + 1 entradas
                    weights = randn(layers{i}{1}, layers{i - 1}{1} + 1) * sqrt(2 / (layers{i - 1}{1} + 1));
                end

                model.layers{i} = struct('weights', weights, 'activation', layers{i}{2});
                
                % Inicializar momentos Adam/AdamW en cero
                model.m{i} = zeros(size(weights));
                model.v{i} = zeros(size(weights));
            end
        end

        function model = compile(model, learning_rate, optimizer, loss_function)
            % Configura los hiperparámetros del entrenamiento
            % learning_rate: tasa de aprendizaje
            % optimizer: 'sgd', 'adam', o 'adamW'
            % loss_function: 'mse' o 'cross_entropy'
            
            model.learning_rate = learning_rate;
            model.optimizer = optimizer;
            model.loss_function = loss_function;
            
            if strcmp(optimizer, 'adam')
                model.beta1 = 0.9;      % Decaimiento media móvil gradiente
                model.beta2 = 0.999;    % Decaimiento media móvil gradiente²
                model.epsilon = 1e-8;   % Evita división por cero
            elseif strcmp(optimizer, 'adamW')
                model.beta1 = 0.9;
                model.beta2 = 0.999;
                model.epsilon = 1e-8;
                model.weight_decay = 0.01;  % Factor decaimiento L2 de pesos
            end
        end

        function outputs = forward(model, x)
            % Forward propagation - Propagación hacia adelante
            % x: matriz de entrada (batch_size x num_features)
            % outputs: cell array con las activaciones de cada capa
            
            outputs = cell(1, model.num_layers + 1);
            outputs{1} = x;  % Capa de entrada (sin procesar)

            for i = 1 : model.num_layers
                % Añadir columna de unos para el bias: [1, x]
                % Trasponer para multiplicación: weights * entrada'
                % weights: (neuronas x entradas), entrada': (entradas x batch)
                z = model.layers{i}.weights * [ones(size(outputs{i}, 1), 1), outputs{i}]';
                activation_function = model.layers{i}.activation;

                % Aplicar función de activación elemento a elemento
                if strcmp(activation_function, 'sigmoid')
                    a = sigmoid(z);
                elseif strcmp(activation_function, 'tanh')
                    a = tanh(z);
                elseif strcmp(activation_function, 'relu')
                    a = relu(z);
                elseif strcmp(activation_function, 'linear')
                    a = z;  % Identidad
                elseif strcmp(activation_function, 'softmax')
                    a = softmax(z);  % Se aplica por columnas
                end

                % Guardar activación traspuesta: (batch x neuronas)
                outputs{i+1} = a';
            end
        end

        function derivative = activation_derivative(~, layer_activation, x)
            % Calcula la derivada de la función de activación
            % NOTA: Para sigmoid y tanh, x debe ser el valor POST-ACTIVACIÓN
            %       Es decir, x = activation(z), no la pre-activación z
            %       Esto es eficiente porque ya calculamos la activación en forward
            
            switch layer_activation
                case 'sigmoid'
                    % Si x = sigmoid(z), entonces sigmoid'(z) = x * (1 - x)
                    derivative = sigmoid_derivative(x);
                case 'tanh'
                    % Si x = tanh(z), entonces tanh'(z) = 1 - x^2
                    derivative = tanh_derivative(x);
                case 'relu'
                    % ReLU'(x) = 1 si x > 0, 0 en otro caso
                    derivative = relu_derivative(x);
                case 'linear'
                    % Derivada de identidad es 1
                    derivative = ones(size(x));
                case 'softmax'
                    % Para softmax + cross_entropy, el gradiente se simplifica
                    % Se maneja como caso especial en compute_gradients
                    derivative = 1;
            end
        end

        function grad = compute_gradients(model, batch_size, outputs, y)
            % Backpropagation - Retropropagación del error
            % Calcula los gradientes de los pesos respecto a la pérdida
            % batch_size: número de muestras
            % outputs: cell array con activaciones de cada capa (de forward)
            % y: etiquetas objetivo (batch_size x num_outputs)
            % grad: cell array con gradientes para cada capa
            
            delta = cell(1, model.num_layers);  % Errores (deltas) de cada capa
            grad = cell(1, model.num_layers);   % Gradientes de los pesos
            a = outputs{end};                   % Activación de la última capa

            % Gradiente de la última capa
            if strcmp(model.layers{end}.activation, 'softmax')
                % Para softmax + cross_entropy, el gradiente simplificado es (a - y)
                % Sin necesidad de multiplicar por la derivada de softmax
                delta{end} = a - y;
            else
                % Para otras activaciones con MSE:
                % delta = f'(a) * (a - y)
                layer_activation = model.layers{end}.activation;
                derivative = model.activation_derivative(layer_activation, a);
                delta{end} = derivative .* (a - y);
            end

            % Retropropagación hacia capas anteriores
            % delta^l = (delta^{l+1} * W^{l+1}) .* f'(a^l)
            for i = model.num_layers - 1 : -1 : 1
                layer_activation = model.layers{i}.activation;
                a = outputs{i+1};  % Activación de la capa actual
                derivative = model.activation_derivative(layer_activation, a);
                
                % Extraer pesos sin el término de bias (columna 1)
                w = model.layers{i + 1}.weights(:, 2:end);
                
                % delta^l = (delta^{l+1} * w^{l+1}) .* f'(a^l)
                % Dimensiones: (batch x n_{l+1}) * (n_{l+1} x n_l) = (batch x n_l)
                delta{i} = derivative .* (delta{i + 1} * w);
            end

            % Calcular gradientes de los pesos
            % grad^l = (1/batch) * delta^l' * [1, a^{l-1}]
            for i = 1 : model.num_layers
                % delta{i}': (n_l x batch)
                % [ones, outputs{i}]: (batch x (n_{l-1}+1))
                % Resultado: (n_l x (n_{l-1}+1)) - mismo tamaño que weights
                grad{i} = (1 / batch_size) * delta{i}' * [ones(batch_size, 1), outputs{i}];
            end
        end

        function model = update_weights(model, grad)
            % Actualiza los pesos usando el optimizador configurado
            % grad: cell array con gradientes para cada capa
            
            for i = 1 : model.num_layers
                if strcmp(model.optimizer, 'sgd')
                    model = sgd(model, grad, i);
                elseif strcmp(model.optimizer, 'adam')
                    model = adam(model, grad, i);
                elseif strcmp(model.optimizer, 'adamW')
                    model = adamW(model, grad, i);
                end
            end
        end

        function [model, history] = train(model, X, Y, epochs)
            % Entrena la red neuronal
            % X: datos de entrada (batch_size x num_features)
            % Y: etiquetas (batch_size x num_outputs)
            % epochs: número de épocas
            % history: vector con la pérdida en cada época
            
            model.t = 0;  % Reiniciar contador de iteraciones
            history = zeros(epochs, 1);
            [batch_size, ~] = size(X);
            [~, num_outputs] = size(Y);

            e = 1e-12;  % Evitar log(0) en cross_entropy

            for epoch = 1 : epochs
                % Forward pass
                outputs = forward(model, X);
                a = outputs{end};

                % Calcular pérdida
                if strcmp(model.loss_function, 'mse')
                    % Error Cuadrático Medio: (1/batch*outputs) * sum((pred - real)^2)
                    loss = sum((outputs{end} - Y).^2, 'all') / (batch_size * num_outputs);
                elseif strcmp(model.loss_function, 'cross_entropy')
                    % Entropía cruzada: -(1/batch) * sum(y * log(pred))
                    % Clip valores para evitar log(0)
                    a_clamp = max(min(a, 1-e), e);
                    loss = -sum(Y .*log(a_clamp), 'all') / batch_size;
                end

                % Backward pass y actualización
                grad = compute_gradients(model, batch_size, outputs, Y);
                model = update_weights(model, grad); 
                
                history(epoch) = loss;
            end
        end

        function y_pred = predict(model, X)
            % Realiza predicciones con la red entrenada
            % X: datos de entrada (batch_size x num_features)
            % y_pred: predicciones (batch_size x num_outputs)
            
            outputs = forward(model, X);
            y_pred = outputs{end};
        end
    end
end
