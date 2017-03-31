function [W1, W2, bias1, bias2, error] = train_GD(inputs, outputs, W1, W2, bias1, bias2)

[~,m] = size(inputs);
lambda =0.00001;
lr = 0.1;


iterations = 150000;
error = zeros(iterations,1);

for k = 1:iterations
    
    h1s = W1*inputs + bias1*ones(1,m);
    g1s = sigmoid(h1s);

    h2s = W2*g1s + bias2*ones(1,m);
    g2s = sigmoid(h2s);

    errors = (g2s - outputs);
    error(k) = 0.5*sum(sum(errors.*errors)) + 0.5*lambda*(sum(sum(W1.*W1)) + sum(sum(W2.*W2)) + sum(bias1.*bias1) + sum(bias2.*bias2));
    disp(error(k));
    g2_1s = sigmoid_1(h2s);
    g1_1s = sigmoid_1(h1s);

    gradW2 = (errors.*g2_1s)*g1s.';
    grad_bias2 = (errors.*g2_1s)*ones(m,1);
    dg1s = ((errors.*g2_1s).'*W2).';
    gradW1 = (dg1s.*g1_1s)*inputs.';
    grad_bias1 = (dg1s.*g1_1s)*ones(m,1);

    gradW2 = gradW2 + lambda*W2;
    gradW1 = gradW1 + lambda*W1;
    grad_bias1 = grad_bias1 + lambda*bias1;
    grad_bias2 = grad_bias2 + lambda*bias2;
    
    W1 = W1 - lr*gradW1;
    W2 = W2 - lr*gradW2;
    bias1 = bias1 - lr*grad_bias1;
    bias2 = bias2 - lr*grad_bias2;
  
end

