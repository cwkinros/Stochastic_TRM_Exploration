function [] = trainNN()

% two layers
% we have w = {W1, W2}, g = {GradW1, GradW2}

images = loadMNISTImages('train-images.idx3-ubyte');
 
% labels: 60000 by 1 matrix (vector) of labels 
labels = loadMNISTLabels('train-labels.idx1-ubyte');
 
% number of inputs k0
%k0 = 784;
start = 300;
 
k0 = 30;
 
m = 150;
inputs = images(start+1:start+k0,1:m);
labels = labels(1:m);

outputs = zeros(10,m);
for i = 1:m
    if (labels(i) == 0)
        labels(i) = 10;
    end
    outputs(labels(i),i) = 1;
end

% make sure to include a bias

n0 = k0; 
n1 = 20;
n2 = 10;

n = n0*n1 + n1*n2 + n1 + n2;

% + 1 for the bias
W1 = rand(n1, n0);
W1 = W1/sum(sum(abs(W1)));
W2 = rand(n2, n1);
W2 = W2/sum(sum(abs(W2)));

bias1 = rand(n1,1);
bias2 = rand(n2,1);

lambda =0.00001;
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

iterations = 100000;
error = zeros(iterations,1);
x = 1:iterations;
gamma = 1;
first_order = false;
use_p0 = true;
options.issym = 1;
options.isreal = 1;
options.maxit = 10000;
for k = 1:iterations
    
    h1s = W1*inputs + bias1*ones(1,m);
    g1s = sigmoid(h1s);
    
    h2s = W2*g1s + bias2*ones(1,m);
    g2s = sigmoid(h2s);
    
    errors = (g2s - outputs);
    error(k) = sum(sum(errors.*errors));
    disp(error(k));
    g2_1s = sigmoid_1(h2s);
    g1_1s = sigmoid_1(h1s);
    
    gradW2 = (errors.*g2_1s)*h1s.';
    grad_bias2 = (errors.*g2_1s)*ones(m,1);
    dg1s = ((errors.*g2_1s).'*W2).';
    gradW1 = (dg1s.*g1_1s)*inputs.';
    grad_bias1 = (dg1s.*g1_1s)*ones(m,1);

    gradW2 = gradW2/m;
    gradW1 = gradW1/m;
    grad_bias1 = grad_bias1 / m;
    grad_bias2 = grad_bias2 / m;
   
    % all the updates
    if first_order
        W1 = W1 - lr*gradW1/m - lambda*W1;
        W2 = W2 - lr*gradW2/m - lambda*W2;
        bias1 = bias1 - lr*grad_bias1 - lambda*bias1;
        bias2 = bias2 - lr*grad_bias2 - lambda*bias2;
    else
        dg2s = errors;
        g2_2s = sigmoid_2(h2s);
        g1_2s = sigmoid_2(h1s);
        g = M1M2_to_m(gradW1,gradW2,grad_bias1,grad_bias2);

        M1 = [zeros(n), eye(n); eye(n), zeros(n)];
        [v,lam,flag] = eigs(@(x)M0x(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs),2*n,-M1,1,'lr',options);
        if flag 
            disp('EIGS DID NOT CONVERGE');
        end
        v = real(v);
        disp('lam:');
        disp(lam);
        p= v(n+1:2*n);
        p = p/sqrt(p.'*p)*gamma;
        [P1_1, P1_2, P1_bias1, P1_bias2] = m_to_M1M2(p,n0,n1,n2);
      
        W1 = W1 + P1_1;
        W2 = W2 + P1_2;
        bias1 = bias1 + P1_bias1;
        bias2 = bias2 + P1_bias2;
        
        sigma = g.'*p + 0.5*p.'*Hv(p,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs);
        % calculate the change in error
        h1s_temp = W1*inputs + bias1*ones(1,m);
        g1s_temp = sigmoid(h1s_temp);

        h2s_temp = W2*g1s_temp + bias2*ones(1,m);
        g2s_temp = sigmoid(h2s_temp);

        errors = (g2s_temp - outputs);
        next_error1 = sum(sum(errors.'*errors));
        if use_p0
            [p0,flag] = pcg(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs),-g,0.001,n*2);
            if flag
                norm_p0 = norm(p0);
            else
                [p0, flag] = cgs(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs),-g,0.01,n*2);
                if flag
                    norm_p0 = norm(p0);
                else
                    norm_p0 = gamma + 1;
                end
            end
            if norm_p0 < gamma
                [P0_1, P0_2, P0_bias1, P0_bias2] = m_to_M1M2(p0,n0,n1,n2);

                W1 = W1 - P1_1 + P0_1;
                W2 = W2 - P1_2 + P0_2;
                bias1 = bias1 - P1_bias1 + P0_bias1;
                bias2 = bias2 - P1_bias2 + P0_bias2;
                h1s_temp = W1*inputs + bias1*ones(1,m);
                g1s_temp = sigmoid(h1s_temp);

                h2s_temp = W2*g1s_temp + bias2*ones(1,m);
                g2s_temp = sigmoid(h2s_temp);

                errors = (g2s_temp - outputs);
                next_error0 = sum(sum(errors.'*errors));
 
                W1 = W1 - P0_1;
                W2 = W2 - P0_2;
                bias1 = bias1 - P0_bias1;
                bias2 = bias2 - P0_bias2;
               
                

                if next_error0 < next_error1
                    disp('a');
                    next_error = next_error0;
                    sigma = g.'*p0 + 0.5*p0.'*Hv(p0,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs);
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
        else
            next_error = next_error1;
            P1 = P1_1;
            P2 = P1_2;
            P_bias1 = P1_bias1;
            P_bias2 = P1_bias2;
        end
        
        
        
        rho = next_error - error(k) / (sigma);
        if rho > lb && sigma < 0
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
print_accuracy(inputs,labels, W1, W2, bias1, bias2);
plot(x,error);
disp('results:');
%disp(bias1);
%disp(bias2);
%disp(W1);
%disp(W1);
    
    