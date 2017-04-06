function [W1, W2, bias1, bias2, error, gammas, rhos, gmag] = train_TRM(inputs, outputs, W1, W2, bias1, bias2, n1,maxiter)

[n0,m] = size(inputs);
[n2,~] = size(outputs);
lambda =0.01;
lr = 0.02;

h1s = zeros(n1,m);
g1s = zeros(n1,m);
h2s = zeros(n2,m);
g2s = zeros(n2,m);

%TRM params
ub = 0.8;
lb = 0.2;
grow = 2.0;
shrink = 0.5;
n = n0*n1 + n1*n2 + n1 + n2;
gamma = 1;

options.issym = 0;
options.isreal = 1;
options.maxit = maxiter;

iterations =100;
error = zeros(iterations,1);
gmag = double(zeros(iterations,1));
rhos = zeros(iterations,1);
gammas = zeros(iterations,1);
prob_next_error = 0;
for k = 1:iterations
    
    [g,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,error(k)] = getG(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
    
    last_error = error(k);
    disp(error(k));
    gmag(k) = sqrt(g.'*g);

    [p1, sigma_p1, next_error1, valid_p1] = getP1(g,gamma,W1,W2,bias1,bias2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,outputs,lambda,false,0,0);
    [p0, sigma_p0, next_error0, valid_p0] = getP0(W1,W2,bias1,bias2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,outputs,lambda,g,gamma,false,0,0);
    
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

    if (gmag(k) < 10^-8)
        break;
    end

end
