function [W1, W2, bias1, bias2,error] = train_TRM_WS(inputs, outputs, W1, W2, bias1, bias2, n1, b, tofile, file, maxiter)

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

sub_tol = 0.001;
sub_maxit = 1000;

tol = 10^-8;
if tofile
    fprintf(file,'TRM_WS: ub=%f, lb=%f, grow=%f, shrink=%f, lambda=%f, n=%d, n0=%d, n1=%d, n2=%d, sub_tol=%f, sub_maxit=%d, tol=%f, b=%d, m=%d\n',ub,lb,grow,shrink,lambda,n,n0,n1,n2,sub_tol,sub_maxit,tol,b,m);
    fprintf(file,'time, time g, time p1, time p0, total error, total gmag, rho, sigma, gamma, step, approx error, approx gmag \n');
end

for k = 1:maxiter
    tic
    
 
    [g_full,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,last_error] = getG(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
    tg = toc;
    indices = randperm(n,b);
    g = zeros(b,1);
    for i=1:b
        g(i) = g_full(indices(i));
    end
    
    t1_pre = toc;
    [p1, sigma_p1, next_error1, valid_p1] = getP1(g,gamma,W1,W2,bias1,bias2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,outputs,lambda,true,indices,b,sub_tol,sub_maxit);
    t1_post = toc; 
    [p0, sigma_p0, next_error0, valid_p0] = getP0(W1,W2,bias1,bias2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,outputs,lambda,g,gamma,true,indices,b,sub_tol,sub_maxit);
    t0_post = toc;
    step = 2; % 2 indicates no valid step, 1 indicates p1, 0 indicates p0
    
 
    if valid_p1 && valid_p0
        if next_error1 < next_error0
            step = 1;
        else
            step = 0;
        end
    else
        if valid_p1
            step = 1;
        else if valid_p0
                step = 0;
            end
        end
    end
    
    if step == 2
        gamma = gamma*shrink;
        continue;
    else if step == 1
            p = p1;
            sigma = sigma_p1;
            next_error = next_error1;
        else
            p = p0;
            sigma = sigma_p0;
            next_error = next_error0;
        end
    end
        
    rho = (next_error - last_error) / sigma;         
    
    if rho > lb && sigma < 0
        [W1,W2,bias1,bias2] = addP(p,n0,n1,n2,W1,W2,bias1,bias2);
        if rho > ub
            gamma = gamma*grow;
        end
    else
        gamma = gamma*shrink;
    end

    if (sqrt(g.'*g) < tol)
        break;
    end
    
    t = toc;
    
    if tofile
        fprintf(file,'%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f \n', t, tg, t1_post - t1_pre, t0_post - t1_post, last_error, sqrt(g.'*g), rho, sigma, gamma, step, last_error, sqrt(g.'*g));
    else
        disp(last_error);
    end
end
