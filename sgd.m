function model = sgd(model, grad, i)
model.layers{i}.weights = model.layers{i}.weights - model.learning_rate * grad{i};