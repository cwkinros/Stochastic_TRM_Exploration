function [W1, W2, bias1, bias2,error] = train_TRM_WS(inputs, outputs, W1, W2, bias1, bias2, n1, b)

[n0,m] = size(inputs);
[n2,~] = size(outputs);
lambda =0.0000001;
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
shrink = 0.5;
n = n0*n1 + n1*n2 + n1 + n2;
gamma = 1;

options.issym = 1;
options.isreal = 1;
options.maxit = 10;

iterations = 15000;
error = zeros(iterations,1);


for k = 1:iterations
    indices = randperm(n,b);

    [g_full,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,error(k)] = getG(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
    
    disp(error(k));
    g = zeros(b,1);
    for i=1:b
        g(i) = g_full(indices(i));
    end

    M1 = [zeros(b), eye(b); eye(b), zeros(b)];
    converge = true;
    try
        [v,lam,flag] = eigs(@(x)M0x_WS(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, indices,lambda),2*b,-M1,1,'lr',options);
    catch
        disp('did not converge');
        converge = false;
    end
    if converge
        if flag 
            disp('EIGS DID NOT CONVERGE');
        end
        v = real(v);
        disp('lam:');
        disp(lam);
        small_p = - (gamma^2)*v(1:b) / (g.'*v(b+1:2*b));
        p = zeros(n,1);
        for i = 1:b
            p(indices(i)) = small_p(i);
        end
        [P1_1, P1_2, P1_bias1, P1_bias2] = m_to_M1M2(p,n0,n1,n2);

        sigma = g.'*small_p + 0.5*small_p.'*Hv_WS(small_p,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda,indices,b);
        % calculate the change in error
      
        next_error1 = getError(W1+P1_1,W2+P1_2,bias1+P1_bias1,bias2+P1_bias2,inputs,outputs,lambda,m);
        
        [p0,flag] = minres(@(v)Hv_WS(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda, indices,b),-g,0.000001,10000);
        if flag == 0
            disp('p0 is valid');
            norm_p0 = sqrt(p0.'*p0);
        else
            disp('p0 is not valid');
            norm_p0 = gamma + 1;
        end
        if norm_p0 < gamma
            full_p0 = zeros(n,1);
            for i = 1:b
                full_p0(indices(i)) = p0(i);
            end
            [P0_1, P0_2, P0_bias1, P0_bias2] = m_to_M1M2(full_p0,n0,n1,n2);

            W1 = W1 + P0_1;
            W2 = W2 + P0_2;
            bias1 = bias1 + P0_bias1;
            bias2 = bias2 + P0_bias2;
            h1s_temp = W1*inputs + bias1*ones(1,m);
            g1s_temp = sigmoid(h1s_temp);

            h2s_temp = W2*g1s_temp + bias2*ones(1,m);
            g2s_temp = sigmoid(h2s_temp);

            errors = (g2s_temp - outputs);
            next_error0 = 0.5*sum(sum(errors.*errors))+0.5*lambda*(sum(sum(W1.*W1)) + sum(sum(W2.*W2)) + sum(bias1.*bias1) + sum(bias2.*bias2));

            W1 = W1 - P0_1;
            W2 = W2 - P0_2;
            bias1 = bias1 - P0_bias1;
            bias2 = bias2 - P0_bias2;

            if next_error0 < next_error1
                disp('a');
                next_error = next_error0;
                sigma = g.'*p0 + 0.5*p0.'*Hv_WS(p0,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda,indices,b);
                P1 = P0_1;
                P2 = P0_2;
                P_bias1 = P0_bias1;
                P_bias2 = P0_bias2;
            else
                disp('b');
                next_error = next_error1;
                P1 = P1_1;
                P2 = P1_2;
                P_bias1 = P1_bias1;
                P_bias2 = P1_bias2; 
            end
        else
            disp('c');
            next_error = next_error1;
            P1 = P1_1;
            P2 = P1_2;
            P_bias1 = P1_bias1;
            P_bias2 = P1_bias2;             
        end

        rho = (next_error - error(k)) / (sigma);
        if rho > lb && sigma < 0
            if next_error > error(k)
                disp('check up');
            end
            W1 = W1 + P1;
            W2 = W2 + P2;
            bias1 = bias1 + P_bias1;
            bias2 = bias2 + P_bias2;
            if rho > ub
                gamma = gamma*grow;
            end
        else

            gamma = gamma*shrink;
        end
        
        
        if (gamma < 10^-10)
            break;
        end
    end
end
