function [W1, W2, bias1, bias2,error] = train_BTRM(inputs, outputs, W1, W2, bias1, bias2, n1, b)

[n0,m] = size(inputs);
[n2,~] = size(outputs);
lambda =0.00001;
lr = 0.02;

h1s = zeros(n1,m);
g1s = zeros(n1,m);
h2s = zeros(n2,m);
g2s = zeros(n2,m);

%TRM params
lb = 0.2;
ub = 0.8;
grow = 1.0;
shrink = 0.8;
n = n0*n1 + n1*n2 + n1 + n2;
gamma = 10;

options.issym = 1;
options.isreal = 1;
options.maxit = 40;

iterations = 200;
error = zeros(iterations,1);
for k = 1:iterations
    proceed = true;
    is = randperm(m,b);
    h1s = W1*inputs(:,is) + bias1*ones(1,b);
    g1s = sigmoid(h1s);
    
    h2s = W2*g1s + bias2*ones(1,b);
    g2s = sigmoid(h2s);
    
    errors = (g2s - outputs(:,is));
    error_little = sum(sum(errors.'*errors));
    error(k) = get_full_error(inputs,outputs,W1,W2,bias1,bias2,m);
    disp(error(k));
    g2_1s = sigmoid_1(h2s);
    g1_1s = sigmoid_1(h1s);
    
    gradW2 = (errors.*g2_1s)*h1s.';
    grad_bias2 = (errors.*g2_1s)*ones(b,1);
    dg1s = ((errors.*g2_1s).'*W2).';
    gradW1 = (dg1s.*g1_1s)*inputs(:,is).';
    grad_bias1 = (dg1s.*g1_1s)*ones(b,1);

    gradW2 = gradW2/m;
    gradW1 = gradW1/m;
    grad_bias1 = grad_bias1 / m;
    grad_bias2 = grad_bias2 / m;
   
    % all the updates
    
    dg2s = errors;
    g2_2s = sigmoid_2(h2s);
    g1_2s = sigmoid_2(h1s);
    g = M1M2_to_m(gradW1,gradW2,grad_bias1,grad_bias2);

    M1 = [zeros(n), eye(n); eye(n), zeros(n)];
    try
        [v,lam,flag] = eigs(@(x)M0x(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is)),2*n,-M1,1,'lr',options);
    catch 
        disp('no good dir');
        
        proceed = false;
    end

    if proceed
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

        sigma = g.'*p + 0.5*p.'*Hv(p,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is));
        % calculate the change in error
        h1s_temp = W1*inputs(:,is) + bias1*ones(1,b);
        g1s_temp = sigmoid(h1s_temp);

        h2s_temp = W2*g1s_temp + bias2*ones(1,b);
        g2s_temp = sigmoid(h2s_temp);

        errors = (g2s_temp - outputs(:,is));
        next_error1 = sum(sum(errors.'*errors));



        next_error = next_error1;
        P1 = P1_1;
        P2 = P1_2;
        P_bias1 = P1_bias1;
        P_bias2 = P1_bias2;             
        
        rho = (next_error - error_little) / sigma;
        
        if rho > lb
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
    end
  
end

