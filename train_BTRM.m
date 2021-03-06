function [W1, W2, bias1, bias2,error] = train_BTRM(inputs, outputs, W1, W2, bias1, bias2, n1, b)

[n0,m] = size(inputs);
[n2,~] = size(outputs);
lambda =0;

%TRM params
lb = 0.2;
ub = 0.8;
grow = 1.2;
shrink = 0.8;
n = n0*n1 + n1*n2 + n1 + n2;
gamma = 1;

options.issym = 0;
options.isreal = 1;
options.maxit = 1000;

iterations = 20000;
error = zeros(iterations,1);
if b > m
    b = m/2;
end
disp(b);
disp(m);

min_gamma = 10^-20;

for k = 1:iterations
    proceed = true;
    is = randperm(m,b);

    
    [g,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,error_little] = getG(W1,W2,bias1,bias2,inputs(:,is),outputs(:,is),lambda,b);

    error(k) = getError(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
    disp(error(k));
    M1 = [zeros(n), eye(n); eye(n), zeros(n)];
    try
        [v,lam,flag] = eigs(@(x)M0x(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is),lambda),2*n,-M1,1,'lr',options);
    catch 
        disp('no good dir');
        
        proceed = false;
    end

    if proceed 
        if flag 
            disp('EIGS DID NOT CONVERGE');
        end
        v = real(v);
        p = - (gamma^2)*v(1:n) / (g.'*v(n+1:2*n));
        if (lam < 0)
            p = p*0;
        end
   
        [P1_1, P1_2, P1_bias1, P1_bias2] = m_to_M1M2(p,n0,n1,n2);


        sigma = g.'*p + 0.5*p.'*Hv(p,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is),lambda);
        if isnan(sigma) || sigma == 0
            disp('here');
        end

               
        next_error1 = getError(W1+P1_1,W2+P1_2,bias1+P1_bias1,bias2+P1_bias2,inputs(:,is),outputs(:,is),lambda,b);
        
        norm_p0 = gamma + 1;
        if sigma > -10^-4 % indicates that it's almost Positive definite
            [p0, flag] = pcg(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is), lambda),-g,0.001,10*n);
            if flag == 0
                disp('p0 is valid, norm:');
                norm_p0 = sqrt(p0.'*p0);
                disp(norm_p0);
            else
                [p0, flag] = lsqr(@(v,word)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is), lambda),-g,0.001,10*n);
                if flag == 0
                    disp('p0 is valid, norm:');
                    norm_p0 = sqrt(p0.'*p0);
                    disp(norm_p0);
                end
            end
        end
            
        %[p0,flag] = cgs(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,0.000001,100000*n);
        %[p0, flag] = pcg(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,0.001,10*n);
        %[p0, flag] = lsqr(@(v,word)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,0.01,10*n);
       
        if norm_p0 < gamma
            [P0_1, P0_2, P0_bias1, P0_bias2] = m_to_M1M2(p0,n0,n1,n2);
            
            next_error0 = getError(W1+P0_1,W2+P0_2,bias1+P0_bias1,bias2+P0_bias2,inputs(:,is),outputs(:,is),lambda,b);
            disp('errors: 0 then 1');
            disp(next_error0);
            disp(next_error1);
            if next_error0 <= next_error1
                disp('a');
                next_error = next_error0;
                sigma = g.'*p0 + 0.5*p0.'*Hv(p0,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs(:,is),lambda);
                if isnan(sigma) || sigma == 0
                    disp('hello');
                end
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
        

      
        if sigma ~= 0
            rho = (next_error - error_little) / (sigma);
        else
            rho = 0;
        end
        if isnan(rho)
            disp('check it out here');
        end
        rhos(k) = rho;
       
        if rho > lb && sigma < 0
            W1 = W1 + P1;
            W2 = W2 + P2;
            bias1 = bias1 + P_bias1;
            bias2 = bias2 + P_bias2;
            gamma = gamma*shrink;
            if rho > ub
                gamma = gamma*grow;
            end
        else
            gamma = gamma*shrink;
        end
       
        if (sqrt(g.'*g) < 10^-8)
            break;
        end
    else
        gamma = gamma*shrink;
    end
    
    if gamma < min_gamma
        gamma = min_gamma
    end
  
end

