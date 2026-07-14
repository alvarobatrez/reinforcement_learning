close all; clear, clc

X = [0 0; 0 1; 1 0; 1 1];
Y = [1; 0; 0; 1];

[~, num_inputs] = size(X);

learning_rate = 0.001;
optimizer = 'adam';
loss_function = 'mse';
epochs = 1000;

layers = {{10, 'relu'} {10, 'relu'} {1, 'sigmoid'}};

model = NeuralNetwork(num_inputs, layers);
model = model.compile(learning_rate, optimizer, loss_function);
[model, history] = model.train(X, Y, epochs);
y_pred = model.predict(X);

disp('Resultados')
for i = 1 : 4
    fprintf('Entrada: [%d %d], Salida: %.2f\n',X(i,1), X(i,2), y_pred(i))  
end

plot(1:epochs, history), title('Función Costo'), grid on
xlabel('Épocas'), ylabel('Error (MSE)')