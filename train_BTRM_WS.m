function [W1, W2, bias1, bias2,error] = train_BTRM_WS(inputs, outputs, W1, W2, bias1, bias2, n1, b_m, b_w)

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
grow = 1.1;
shrink = 0.8;
n = n0*n1 + n1*n2 + n1 + n2;
gamma = 10;

options.issym = 1;
options.isreal = 1;
options.maxit = 50;

iterations = 2000;
error = zeros(iterations,1);
if b_m > m
    b_m = m/2;
end
for k = 1:iterations
    proceed = true;
    indices = randperm(n,b_w);
    is = randperm(m,b_m);
    h1s = W1*inputs(:,is) + bias1*ones(1,b_m);
    g1s = sigmoid(h1s);
    
    h2s = W2*g1s + bias2*ones(1,b_m);
    g2s = sigmoid(h2s);
    
    errors = (g2s - outputs(:,is));
    error_little = sum(sum(errors.'*errors));
    error(k) = get_full_error(inputs,outputs,W1,W2,bias1,bias2,m);
    disp(error(k));
    g2_1s = sigmoid_1(h2s);
    g1_1s = sigmoid_1(h1s);
    
    gradW2 = (errors.*g2_1s)*g1s.';
    grad_bias2 = (errors.*g2_1s)*ones(b_m,1);
    dg1s = ((errors.*g2_1s).'*W2).';
    gradW1 = (dg1s.*g1_1s)*inputs(:,is).';
    grad_bias1 = (dg1s.*g1_1s)*ones(b_m,1);

    gradW2 = gradW2/m + lambda*W2;
    gradW1 = gradW1/m + lambda*W1;
    grad_bias1 = grad_bias1 / m + lambda*bias1;
    grad_bias2 = grad_bias2 / m + lambda*bias2;
   
    % all the updates
    
    dg2s = errors;
    g2_2s = sigmoid_2(h2s);
    g1_2s = sigmoid_2(h1s);
    g_full = M1M2_to_m(gradW1,gradW2,grad_bias1,grad_bias2);
    
    g = zeros(b_w,1);
    for i=1:b_w
        g(i) = g_full(indices(i));
    end

    M1 = [zeros(b_w), eye(b_w); eye(b_w), zeros(b_w)];
    try
        [v,lam,flag] = eigs(@(x)M0x_WS(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is),indices, lambda),2*b_w,-M1,1,'lr',options);
    catch 
        disp('no good dir');
        
        proceed = false;
    end

    if proceed
        if flag 
            disp('EIGS DID NOT CONVERGE');
        end
        v = real(v);
        %disp('lam:');
        %disp(lam);
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

        sigma = g.'*small_p + 0.5*small_p.'*Hv_WS(small_p,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is),lambda,indices,b_w);
        % calculate the change in error
        h1s_temp = W1*inputs(:,is) + bias1*ones(1,b_m);
        g1s_temp = sigmoid(h1s_temp);

        h2s_temp = W2*g1s_temp + bias2*ones(1,b_m);
        g2s_temp = sigmoid(h2s_temp);

        errors = (g2s_temp - outputs(:,is));
        next_error_little1 = sum(sum(errors.'*errors));
        W1 = W1 - P1_1;
        W2 = W2 - P1_2;
        bias1 = bias1 - P1_bias1;
        bias2 = bias2 - P1_bias2;
        
        [p0,flag] = pcg(@(v)Hv_WS(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is),lambda,indices,b_w),-g,0.001,b_w*2);
        if flag
            norm_p0 = norm(p0);
        else
            [p0, flag] = cgs(@(v)Hv_WS(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is),lambda,indices,b_w),-g,0.01,b_w*2);
            if flag
                norm_p0 = norm(p0);
            else
                norm_p0 = gamma + 1;
            end
        end
        if norm_p0 < gamma
            full_p0 = zeros(n,1);
            for i = 1:b_w
                full_p0(indices(i)) = p0(i);
            end
            [P0_1, P0_2, P0_bias1, P0_bias2] = m_to_M1M2(full_p0,n0,n1,n2);

            W1 = W1 + P0_1;
            W2 = W2 + P0_2;
            bias1 = bias1 + P0_bias1;
            bias2 = bias2 + P0_bias2;
            h1s_temp = W1*inputs(:,is) + bias1*ones(1,b_m);
            g1s_temp = sigmoid(h1s_temp);

            h2s_temp = W2*g1s_temp + bias2*ones(1,b_m);
            g2s_temp = sigmoid(h2s_temp);

            errors = (g2s_temp - outputs(:,is));
            next_error_little0 = sum(sum(errors.'*errors));

            W1 = W1 - P0_1;
            W2 = W2 - P0_2;
            bias1 = bias1 - P0_bias1;
            bias2 = bias2 - P0_bias2;

            if next_error_little0 < next_error_little1
            %    disp('a');
                next_error = next_error_little0;
                sigma = g.'*p0 + 0.5*p0.'*Hv_WS(p0,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is),lambda,indices,b_w);
                P1 = P0_1;
                P2 = P0_2;
                P_bias1 = P0_bias1;
                P_bias2 = P0_bias2;
            else
          %      disp('b');
                next_error = next_error_little1;
                P1 = P1_1;
                P2 = P1_2;
                P_bias1 = P1_bias1;
                P_bias2 = P1_bias2; 
            end
        else
         %   disp('c');
            next_error = next_error_little1;
            P1 = P1_1;
            P2 = P1_2;
            P_bias1 = P1_bias1;
            P_bias2 = P1_bias2;             
        end

        rho = (next_error - error_little) / (sigma);
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

