function [W1, W2, bias1, bias2,error] = train_MBTRM_WS(inputs, outputs, W1, W2, bias1, bias2, n1, b_m, b_w)

[n0,m] = size(inputs);
[n2,~] = size(outputs);
lambda =0.00001;
%lambda = 0;
lr = 0.02;

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
gamma = 1;

options.issym = 0;
options.isreal = 1;
options.maxit = 100000;

iterations = 10000;
error = zeros(iterations,1);
for k = 1:iterations
    indices = randperm(n,b_w);
    ims = randperm(m,b_m);
   
    % all the updates
    [g_full,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,~] = getG(W1,W2,bias1,bias2,inputs(:,ims),outputs(:,ims),lambda,b_m);
    error(k) = getError(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
    disp(error(k));
    g = zeros(b_w,1);
    for i=1:b_w
        g(i) = g_full(indices(i));
    end

    M1 = [zeros(b_w), eye(b_w); eye(b_w), zeros(b_w)];
    converge = true;
    try
        [v,lam,flag] = eigs(@(x)M0x_WS(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,ims), indices,lambda),2*b_w,-M1,1,'lr',options);
    catch
        disp('did not converge');
        converge = false;
    end
    if converge
        v = real(v);
        disp('lam:');
        disp(lam);
        small_p = - (gamma^2)*v(1:b_w) / (g.'*v(b_w+1:2*b_w));
        p = zeros(n,1);
        for i = 1:b_w
            p(indices(i)) = small_p(i);
        end
        [P1_1, P1_2, P1_bias1, P1_bias2] = m_to_M1M2(p,n0,n1,n2);

        W1 = W1 + P1_1;
        W2 = W2 + P1_2;
        bias1 = bias1 + P1_bias1;
        bias2 = bias2 + P1_bias2;
        
        
    end
    gamma = gamma*shrink;        
        
          
    if (gamma < 10^-10)
        break;
    end
end