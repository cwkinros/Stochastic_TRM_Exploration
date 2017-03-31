function [W1, W2, bias1, bias2,error] = train_STRM_WS(inputs, outputs, W1, W2, bias1, bias2, n1, b)

[n0,m] = size(inputs);
[n2,~] = size(outputs);
lambda =0.0001;
%lambda = 0;
a = 5000;
b = 5000;


h1s = zeros(n1,m);
g1s = zeros(n1,m);
h2s = zeros(n2,m);
g2s = zeros(n2,m);

%TRM params
ub = 0.8;
lb = 0.2;
grow = 1.5;
shrink = 0.999;
n = n0*n1 + n1*n2 + n1 + n2;
gamma = a/b;

options.issym = 0;
options.isreal = 1;
options.maxit = 1000;

iterations = 10000;
error = zeros(iterations,1);
for k = 1:iterations
    indices = randperm(n,b);
    i = randi(m);
    
    [g_full,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,~] = getG(W1,W2,bias1,bias2,inputs(:,i),outputs(:,i),lambda,1);
    
    error(k) = getError(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
    disp(error(k));
    
    g = zeros(b,1);
    for i=1:b
        g(i) = g_full(indices(i));
    end

    M1 = [zeros(b), eye(b); eye(b), zeros(b)];
    converge = true;
    try
        [v,lam,flag] = eigs(@(x)M0x_WS(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,i), indices,lambda),2*b,-M1,1,'lr',options);
    catch
        disp('did not converge');
        converge = false;
    end
    if converge
        v = real(v);
        small_p = - (gamma^2)*v(1:b) / (g.'*v(b+1:2*b));
        p = zeros(n,1);
        for i = 1:b
            p(indices(i)) = small_p(i);
        end
        [P1_1, P1_2, P1_bias1, P1_bias2] = m_to_M1M2(p,n0,n1,n2);

        W1 = W1 + P1_1;
        W2 = W2 + P1_2;
        bias1 = bias1 + P1_bias1;
        bias2 = bias2 + P1_bias2;
        
        
    end
    gamma = a / (b + k);        
        
          
    if (gamma < 10^-10)
        break;
    end
end