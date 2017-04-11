function [p1,sigma, next_error1, converged] = getP1(g,gamma,W1,W2,bias1,bias2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,outputs,lambda,WS,indices,b,tol,maxiter)


options.maxit = maxiter;
options.isreal = 1;
options.issym = 0;
options.tol = tol;

[n1,n0] = size(W1);
[n2,~] = size(W2);
n = n2*(n1 + 1) + n1*(n0 + 1);
[~,m] = size(inputs);
converged = true;
try
    if WS
        M1 = [zeros(b), eye(b); eye(b), zeros(b)];
        [v,~,flag] = eigs(@(x)M0x_WS(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, indices,lambda),2*b,-M1,1,'lr',options);
    else
        M1 = [zeros(n), eye(n); eye(n), zeros(n)];
        [v,~,flag] = eigs(@(x)M0x(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda),2*n,-M1,1,'lr',options);
    end
    v = real(v);
    if flag > 0
        converged = false;
    end
catch
    converged = false;
end

if converged
    if WS
        small_p = - (gamma^2)*v(1:b) / (g.'*v(b+1:2*b));
        p1 = zeros(n,1);
        for i = 1:b
            p1(indices(i)) = small_p(i);
        end 
        sigma = g.'*small_p + 0.5*small_p.'*Hv_WS(small_p,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda,indices,b);
        
    else
        p1 = - (gamma^2)*v(1:n) / (g.'*v(n+1:2*n));
        sigma = g.'*p1 + 0.5*p1.'*Hv(p1,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda);
    end
    [P1_1, P1_2, P1_bias1, P1_bias2] = m_to_M1M2(p1,n0,n1,n2);
    next_error1 = getError(W1+P1_1,W2+P1_2,bias1+P1_bias1,bias2+P1_bias2,inputs,outputs,lambda,m);

else
    p1 = zeros(n,1);
    sigma = 0;
    next_error1 = -1;
end
        