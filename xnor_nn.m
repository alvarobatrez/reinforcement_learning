close all; clear, clc

X = [0 0; 0 1; 1 0; 1 1];
Y = [1; 0; 0; 1];

[~, num_inputs] = size(X);
 
learning_rate = 0.01;
optimizer = 'sgdm';
loss_function = 'mse';
epochs = 1000;

layers = [
    featureInputLayer(num_inputs)
    fullyConnectedLayer(10)
    reluLayer
    fullyConnectedLayer(10)
    reluLayer
    fullyConnectedLayer(1)
    sigmoidLayer
    ];

net = dlnetwork(layers);
options = trainingOptions(optimizer, MaxEpochs=epochs, InitialLearnRate=learning_rate, Plots='training-progress');
net = trainnet(X, Y, net, loss_function, options);
y_pred = net.predict(X);

disp('Resultados')
for i = 1 : 4
    fprintf('Entrada: [%d %d], Salida: %.2f\n',X(i,1), X(i,2), y_pred(i))  
end