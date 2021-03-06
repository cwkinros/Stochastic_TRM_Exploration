function [W1, W2, bias1, bias2, error] = train_SGD_TRM(inputs, outputs, W1, W2, bias1, bias2)

[n1,n0] = size(W1);
[n2,~] = size(W2);

n = n0*n1 + n1 + n1*n2 + n2;
[~,m] = size(inputs);
lambda =0.00001;
a = 10000;
b = 5000;

iterations = 30000;
error = zeros(iterations,1);
min = inf;
iter_since_last_min = 0;
thresh = 100;
gamma = 1;
options.issym = 0;
options.isreal = 1;
options.maxit = 20;

for k = 1:iterations
    lr = a / (b + k);
    iter_since_last_min = iter_since_last_min + 1;
    i = randi(m);
    h1s = W1*inputs(:,i) + bias1;
    g1s = sigmoid(h1s);

    h2s = W2*g1s + bias2;
    g2s = sigmoid(h2s);

    errors = (g2s - outputs(:,i));
 
    error(k) = getError(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
    if (error(k) < min)
        min = error(k);
        iter_since_last_min = 0;
    end
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
    
    if iter_since_last_min > thresh
        g = M1M2_to_m(gradW1,gradW2,grad_bias1,grad_bias2);
        
        
        next_error = error(k) + 1;
        gamma = 2;
        while next_error > error(k)
            gamma = gamma/2;
            proceed = true;
            try
                M1 = [zeros(n), eye(n); eye(n), zeros(n)];
                [g,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,~] = getG(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
                [v,lam,flag] = eigs(@(x)M0x(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda),2*n,-M1,1,'lr',options);
            catch 
                disp('no good dir');

                proceed = false;

            end      
            if proceed
                disp('TRM');
                v = real(v);
                p = - (gamma^2)*v(1:n) / (g.'*v(n+1:2*n));
                [P1_1, P1_2, P1_bias1, P1_bias2] = m_to_M1M2(p,n0,n1,n2);
                next_error = getError(W1+P1_1,W2+P1_2,bias1+P1_bias1,bias2+P1_bias2,inputs,outputs,lambda,m);
            end
        end
        
        W1 = W1 + P1_1;
        W2 = W2 + P1_2;
        bias1 = bias1 + P1_bias1;
        bias2 = bias2 + P1_bias2;
    else
        W1 = W1 - lr*gradW1;
        W2 = W2 - lr*gradW2;
        bias1 = bias1 - lr*grad_bias1;
        bias2 = bias2 - lr*grad_bias2;
    end    

end

plot(error);
legend('SGD','SGD + TRM');