function [W1, W2, bias1, bias2, error, gammas, rhos, gmag] = train_TRM(inputs, outputs, W1, W2, bias1, bias2, n1,maxiter)

[n0,m] = size(inputs);
[n2,~] = size(outputs);
lambda =0.00001;
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

iterations = 50;
error = zeros(iterations,1);
gmag = zeros(iterations,1);
rhos = zeros(iterations,1);
gammas = zeros(iterations,1);
prob_next_error = 0;
for k = 1:iterations
    
    [g,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,error(k)] = getG(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
    last_error = error(k);
    disp(error(k));
    gmag(k) = sqrt(g.'*g);
    
    M1 = [zeros(n), eye(n); eye(n), zeros(n)];
    converged = true;
    try
        disp('start eigs');
        [v,lam,flag] = eigs(@(x)M0x(x,g,gamma,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda),2*n,-M1,1,'lr',options);
        disp('end eigs');
    catch
        converged = false;
    end
    
    if converged 
        if flag 
            disp('EIGS DID NOT CONVERGE');
        end
        v = real(v);
        disp('lam:');
        disp(lam);



        %p = (p/sqrt(p.'*p))*gamma;
        
        p = - (gamma^2)*v(1:n) / (g.'*v(n+1:2*n));
        if (lam < 0)
            p = p*0;
        end
        disp(sqrt(p.'*p));
        %p = - (gamma^2)*v(1:n) / (g.'*v(n+1:2*n));
        [P1_1, P1_2, P1_bias1, P1_bias2] = m_to_M1M2(p,n0,n1,n2);


        sigma = g.'*p + 0.5*p.'*Hv(p,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda);
        if isnan(sigma) || sigma == 0
            disp('here');
        end

               
        next_error1 = getError(W1+P1_1,W2+P1_2,bias1+P1_bias1,bias2+P1_bias2,inputs,outputs,lambda,m);
        
        [p0,flag] = cgs(@(v)Hv(v,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs, lambda),-g,0.5,n);
        if flag == 0
            disp('p0 is valid');
            norm_p0 = sqrt(p0.'*p0);
        else
            disp('p0 is not valid');
            norm_p0 = gamma + 1;
        end
        if norm_p0 < gamma
            [P0_1, P0_2, P0_bias1, P0_bias2] = m_to_M1M2(p0,n0,n1,n2);
            
            next_error0 = getError(W1+P0_1,W2+P0_2,bias1+P0_bias1,bias2+P0_bias2,inputs,outputs,lambda,m);


            if next_error0 < next_error1
                disp('a');
                next_error = next_error0;
                sigma = g.'*p0 + 0.5*p0.'*Hv(p0,W1,W2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,inputs,lambda);
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
        

        gammas(k) = gamma;
        if sigma ~= 0
            rho = (next_error - last_error) / (sigma);
        else
            rho = 1;
        end
        if isnan(rho)
            disp('check it out here');
        end
        rhos(k) = rho;
       
        if rho > lb && sigma < 0
            %W1 = W1 + P1;
            %W2 = W2 + P2;
            %bias1 = bias1 + P_bias1;
            %bias2 = bias2 + P_bias2;
            gamma = gamma*shrink;
            %if rho > ub
            %    gamma = gamma*grow;
            %end
        else
            gamma = gamma*shrink;
        end
       
        if (gamma < 10^-10)
            break;
        end
    else
        gamma = gamma*shrink;
    end
end

plot(gammas);
hold on;
plot(error);
hold on;
plot(rhos);
hold on;
plot(gmag);
legend('gamma','error','rhos','gmag');