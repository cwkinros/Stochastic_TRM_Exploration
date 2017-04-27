function [p0, sigma, next_error0, valid] = getP0(W1,W2,bias1,bias2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,outputs, lambda,g, gamma, WS, indices, b, tol, maxit)

% valid refers to a P0 solution where it reached convergence AND it's
% magnitude < gamma

%[p0,flag] = minres(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,0.01,10*n);
%[p0,flag] = cgs(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,0.000001,100000*n);
%[p0, flag] = pcg(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,0.001,10*n);
%[p0, flag] = lsqr(@(v,word)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,0.01,10*n);
[n1,n0] = size(W1);
[n2,~] = size(W2);
[~,m] = size(inputs);
if isnan(bias1)
    bias = false;
    n = n2*n1 + n1*n0;
else
    bias = true;
    n = n2*(n1+1) + n1*(n0+1);
end

if WS
    [p0, flag] = lsqr(@(v,word)Hv_WS(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda,indices,b, bias),-g,tol,maxit);
%    if type == 1
%        [p0, flag] = pcg(@(v)Hv_WS(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda,indices,b),-g,tol,maxit);
%    else if type == 2
%            [p0, flag] = cgs(@(v)Hv_WS(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda,indices,b),-g,tol,maxit);
%        else if type == 3
%                [p0, flag] = minres(@(v)Hv_WS(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda,indices,b),-g,tol,maxit);   
%            else
%                [p0, flag] = lsqr(@(v,word)Hv_WS(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda,indices,b),-g,tol,maxit);
%            end
%        end
%    end
    
else
    [p0, flag] = lsqr(@(v,word)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,tol,maxit);
%    if type == 1        
%        [p0, flag] = pcg(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,tol,maxit);
%    else if type == 2
%            [p0, flag] = cgs(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,tol,maxit);
%        else if type == 3
%                [p0, flag] = minres(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,tol,maxit);
%            else
%                [p0, flag] = lsqr(@(v,word)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,tol,maxit);
%            end
%        end
%    end
end
if sqrt(p0.'*p0) == 0
    flag = 1;
end
valid = false;
sigma = 0;
next_error0 = inf;
if flag == 0
    if sqrt(p0.'*p0) < gamma
        valid = true;
        if WS
            full_p0 = zeros(n,1);
            for i = 1:b
                full_p0(indices(i)) = p0(i);
            end
            sigma = g.'*p0 + 0.5*p0.'*Hv_WS(p0,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda,indices,b,bias);
            [P0_1, P0_2, P0_bias1, P0_bias2] = m_to_M1M2(full_p0,n0,n1,n2);
            p0 = full_p0;
        else
            sigma = g.'*p0 + 0.5*p0.'*Hv(p0,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda);
            [P0_1, P0_2, P0_bias1, P0_bias2] = m_to_M1M2(p0,n0,n1,n2);
        end
        next_error0 = getError(W1 + P0_1, W2+P0_2, bias1 + P0_bias1, bias2 + P0_bias2,inputs,outputs,lambda,m);
    end
end
        
