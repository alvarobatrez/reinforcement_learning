function model = adamW(model, grad, i)
model.t = model.t + 1;
model.m{i} = model.beta1 * model.m{i} + (1 - model.beta1) * grad{i};
model.v{i} = model.beta2 * model.v{i} + (1 - model.beta2) * (grad{i}.^2);
m_hat = model.m{i} / (1 - model.beta1^model.t);
v_hat = model.v{i} / (1 - model.beta2^model.t);
weight_decay_term = model.weight_decay * model.layers{i}.weights;
model.layers{i}.weights = model.layers{i}.weights - model.learning_rate * (m_hat ./ (sqrt(v_hat) + model.epsilon) + weight_decay_term);