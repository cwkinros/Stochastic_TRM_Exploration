function [W1, W2, bias1, bias2,error] = train_STRM_RMI(inputs, outputs, W1, W2, bias1, bias2, n1)

[n0,m] = size(inputs);
[n2,~] = size(outputs);
lambda =0.00001;
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
iterations = 800;
error = zeros(iterations,1);
for k = 1:iterations
    proceed = true;
    options.maxit = randi(10 + k);
    i = randi(m);
    h1s = W1*inputs(:,i) + bias1;
    g1s = sigmoid(h1s);
    
    h2s = W2*g1s + bias2;
    g2s = sigmoid(h2s);
    
    errors = (g2s - outputs(:,i));
    
    error(k) = get_full_error(inputs,outputs,W1,W2,bias1,bias2,m);
    disp(error(k));
    g2_1s = sigmoid_1(h2s);
    g1_1s = sigmoid_1(h1s);
    
    gradW2 = (errors.*g2_1s)*h1s.';
    grad_bias2 = (errors.*g2_1s);
    dg1s = ((errors.*g2_1s).'*W2).';
    gradW1 = (dg1s.*g1_1s)*inputs(:,i).';
    grad_bias1 = (dg1s.*g1_1s);

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
        [v,lam,flag] = eigs(@(x)M0x(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,i)),2*n,-M1,1,'lr',options);
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

        sigma = g.'*p + 0.5*p.'*Hv(p,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,i));
        % calculate the change in error
        h1s_temp = W1*inputs(:,i) + bias1;
        g1s_temp = sigmoid(h1s_temp);

        h2s_temp = W2*g1s_temp + bias2;
        g2s_temp = sigmoid(h2s_temp);

        errors = (g2s_temp - outputs(:,i));
        next_error1 = sum(sum(errors.'*errors));



        next_error = next_error1;
        P1 = P1_1;
        P2 = P1_2;
        P_bias1 = P1_bias1;
        P_bias2 = P1_bias2;             


        W1 = W1 + P1;
        W2 = W2 + P2;
        bias1 = bias1 + P_bias1;
        bias2 = bias2 + P_bias2;
        gamma = gamma*shrink;
    end
  
end

