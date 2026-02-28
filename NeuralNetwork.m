classdef NeuralNetwork

    properties
        layers
        learning_rate
        optimizer
        loss_function
        num_layers
        beta1
        beta2
        epsilon
        weight_decay
        m
        v
        t = 0
    end

    methods
        function model = NeuralNetwork(input_size, layers)
            model.num_layers = length(layers);
            model.m = cell(1, model.num_layers);
            model.v = cell(1, model.num_layers);

            for i = 1 : model.num_layers
                if i == 1
                    weights = randn(layers{i}{1}, input_size + 1) * sqrt(2 / (input_size + 1));
                else
                    weights = randn(layers{i}{1}, layers{i - 1}{1} + 1) * sqrt(2 / (layers{i - 1}{1} + 1));
                end

                model.layers{i} = struct('weights', weights, 'activation', layers{i}{2});
                model.m{i} = zeros(size(weights));
                model.v{i} = zeros(size(weights));
            end
        end

        function model = compile(model, learning_rate, optimizer, loss_function)
            model.learning_rate = learning_rate;
            model.optimizer = optimizer;
            model.loss_function = loss_function;
            if strcmp(optimizer, 'adam')
                model.beta1 = 0.9;
                model.beta2 = 0.999;
                model.epsilon = 1e-8;
            elseif strcmp(optimizer, 'adamW')
                model.beta1 = 0.9;
                model.beta2 = 0.999;
                model.epsilon = 1e-8;
                model.weight_decay = 0.01;
            end
        end

        function outputs = forward(model, x)
            outputs = cell(1, model.num_layers + 1);
            outputs{1} = x;

            for i = 1 : model.num_layers
                z = model.layers{i}.weights * [ones(size(outputs{i}, 1), 1), outputs{i}]';
                activation_function = model.layers{i}.activation;

                if strcmp(activation_function, 'sigmoid')
                    a = sigmoid(z);
                elseif strcmp(activation_function, 'tanh')
                    a = tanh(z);
                elseif strcmp(activation_function, 'relu')
                    a = relu(z);
                elseif strcmp(activation_function, 'linear')
                    a = z;
                elseif strcmp(activation_function, 'softmax')
                    a = softmax(z);
                end

                outputs{i+1} = a';
            end
        end

        function derivative = activation_derivative(~, layer_activation, x)
            switch layer_activation
                case 'sigmoid'
                    derivative = sigmoid_derivative(x);
                case 'tanh'
                    derivative = tanh_derivative(x);
                case 'relu'
                    derivative = relu_derivative(x);
                case 'linear'
                    derivative = ones(size(x));
                case 'softmax'
                    derivative = 1;
            end
        end

        function grad = compute_gradients(model, batch_size, outputs, y)
            delta = cell(1, model.num_layers);
            grad = cell(1, model.num_layers);
            a = outputs{end};

            if strcmp(model.layers{end}.activation, 'softmax')
                delta{end} = a - y;
            else
                layer_activation = model.layers{end}.activation;
                derivative = model.activation_derivative(layer_activation, a);
                delta{end} = derivative .* (a - y);
            end

            for i = model.num_layers - 1 : -1 : 1
                layer_activation = model.layers{i}.activation;
                a = outputs{i+1};
                derivative = model.activation_derivative(layer_activation, a);
                
                w = model.layers{i + 1}.weights(:, 2:end);
                delta{i} = derivative .* (delta{i + 1} * w);
            end

            for i = 1 : model.num_layers
                grad{i} = (1 / batch_size) * delta{i}' * [ones(batch_size, 1), outputs{i}];
            end
        end

        function model = update_weights(model, grad)
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
            model.t = 0;
            history = zeros(epochs, 1);
            [batch_size, ~] = size(X);
            [~, num_outputs] = size(Y);

            e = 1e-12;

            for epoch = 1 : epochs
                loss = 0;
                outputs = forward(model, X);
                a = outputs{end};

                if strcmp(model.loss_function, 'mse')
                    loss = loss + sum((outputs{end} - Y).^2, 'all') / (batch_size * num_outputs);
                elseif strcmp(model.loss_function, 'cross_entropy')
                    a_clamp = max(min(a, 1-e), e);
                    loss = -sum(Y .*log(a_clamp), 'all') / batch_size;
                end

                grad = compute_gradients(model, batch_size, outputs, Y);
                model = update_weights(model, grad); 
                
                history(epoch) = loss;
            end
        end

        function y_pred = predict(model, X)
            outputs = forward(model, X);
            y_pred = outputs{end};
        end
    end
end