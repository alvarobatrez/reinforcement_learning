function model = sgd(model, grad, i)
    % Stochastic Gradient Descent (SGD)
    % Actualización: w = w - learning_rate * gradiente
    %
    % Parámetros:
    %   model: instancia de NeuralNetwork
    %   grad: cell array con gradientes
    %   i: índice de la capa a actualizar
    
    model.layers{i}.weights = model.layers{i}.weights - model.learning_rate * grad{i};
end
