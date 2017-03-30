function [W1, W2, bias1, bias2, error] = train_SGD_TRM(inputs, outputs, W1, W2, bias1, bias2)

[~,m] = size(inputs);
lambda =0.00001;
a = 10000;
b = 5000;

iterations = 50000;
error = zeros(iterations,1);

for k = 1:iterations
    lr = a / (b + k);
    i = randi(m);
    h1s = W1*inputs(:,i) + bias1;
    g1s = sigmoid(h1s);

    h2s = W2*g1s + bias2;
    g2s = sigmoid(h2s);

    errors = (g2s - outputs(:,i));
    error(k) = getError(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
    
    disp(error(k));
    g2_1s = sigmoid_1(h2s);
    g1_1s = sigmoid_1(h1s);

    gradW2 = (errors.*g2_1s)*g1s.';
    grad_bias2 = (errors.*g2_1s);
    dg1s = ((errors.*g2_1s).'*W2).';
    gradW1 = (dg1s.*g1_1s)*inputs(:,i).';
    grad_bias1 = (dg1s.*g1_1s);

    gradW2 = gradW2 + lambda*W2;
    gradW1 = gradW1 + lambda*W1;
    grad_bias1 = grad_bias1 + lambda*bias1;
    grad_bias2 = grad_bias2 + lambda*bias2;
    
    W1 = W1 - lr*gradW1;
    W2 = W2 - lr*gradW2;
    bias1 = bias1 - lr*grad_bias1;
    bias2 = bias2 - lr*grad_bias2;
    
end