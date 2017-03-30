function [W1, W2, bias1, bias2,error] = train_MBTRM(inputs, outputs, W1, W2, bias1, bias2, n1, b)

[n0,m] = size(inputs);
[n2,~] = size(outputs);
lambda =0.000001;
lr = 0.02;

h1s = zeros(n1,m);
g1s = zeros(n1,m);
h2s = zeros(n2,m);
g2s = zeros(n2,m);

%TRM params

shrink = 0.99;
n = n0*n1 + n1*n2 + n1 + n2;
gamma = 1;

options.issym = 1;
options.isreal = 1;
options.maxit = 20;

iterations = 800;
error = zeros(iterations,1);
for k = 1:iterations
    proceed = true;
    is = randperm(m,b);
    [g,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,~] = getG(W1,W2,bias1,bias2,inputs(:,is),outputs(:,is),lambda,b);
    error(k) = getError(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
    disp(error(k));
    M1 = [zeros(n), eye(n); eye(n), zeros(n)];
    try
        [v,lam,flag] = eigs(@(x)M0x(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is),lambda),2*n,-M1,1,'lr',options);
    catch 
        disp('no good dir');
        
        proceed = false;
    end

    if proceed
        v = real(v);
        disp('lam:');
        disp(lam);
        p = - (gamma^2)*v(1:n) / (g.'*v(n+1:2*n));
        [P1, P2, P_bias1, P_bias2] = m_to_M1M2(p,n0,n1,n2);
           

        W1 = W1 + P1;
        W2 = W2 + P2;
        bias1 = bias1 + P_bias1;
        bias2 = bias2 + P_bias2;
        gamma = gamma*shrink;
    end
  
end

