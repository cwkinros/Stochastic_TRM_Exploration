function [W1, W2, bias1, bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr, decay)

[n0,m] = size(inputs);
[n2,~] = size(outputs);
lambda = 10^(-10);

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


if TRMstep
    b_m = b_m_big;
else
    b_m = b_m_mini;
end
if MS == 1
    num = 1;
else if MS == 2
        num = b_m;
    else
        num = m;
    end
end


sub_tol = 0.001;
sub_maxit = 1000;
tol = 10^-8;
rho = 0;
sigma = 0;
step = 0;
if tofile
    fprintf(file,'TRM WS=%d, MS=%d, TRMstep=%d: ub=%f, lb=%f, grow=%f,shrink=%f, b_w=%d,b_m=%d, lr=%f, decay=%f, lambda=%f, n=%d, n0=%d, n1=%d, n2=%d, sub_tol=%f, sub_maxit=%d, tol=%e, m=%d \n',WS,MS,TRMstep,ub,lb,grow,shrink,b_w,b_m,lr,decay,lambda,n,n0,n1,n2,sub_tol,sub_maxit,tol,m);
    fprintf(file,'time, iter_t, subset_m_time, g_time, subset_w_time, p1_time, t2, t3, total error, total gmag, rho, sigma, gamma, step, approx error, approx gmag \n');
end
rolling_t = 0;
if MS > 0
    gamma = lr;
end
for k = 1:maxiter
    if MS == 1
        [g_total,~,~,~,~,~,~,~,full_error] = getG(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
        tic
        i = randi(m);
        input_set = inputs(:,i);
        output_set = outputs(:,i);
        subset_m_time = toc;
    else if MS == 2
            [g_total,~,~,~,~,~,~,~,full_error] = getG(W1,W2,bias1,bias2,inputs,outputs,lambda,m);
            tic
            is = randperm(m,b_m);
            input_set = inputs(:,is);
            output_set = outputs(:,is);
            subset_m_time = toc;
        else
            input_set = inputs;
            output_set = outputs;
            subset_m_time = 0;
        end
    end
    
    disp('a');     
    tic
    [g_full,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,last_error] = getG(W1,W2,bias1,bias2,input_set,output_set,lambda,num);
    g_time = toc;
    disp('b');

    
    if WS
        tic;
        indices = randperm(n,b_w);
        g = zeros(b_w,1);
        for i=1:b_w
            g(i) = g_full(indices(i));
        end 
        subset_w_time = toc;
    else
        b_w = n;
        g = g_full;
        indices = 0;
        subset_w_time = 0;
    end
    
    disp('c');
    
    if GD
        tic
        p1 = -gamma*g_full;
        p1_time = toc;
    else
        tic
        [p1, sigma_p1, next_error1, valid_p1] = getP1(g,gamma,W1,W2,bias1,bias2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,input_set,output_set,lambda,WS,indices,b_w,sub_tol,sub_maxit);
        p1_time = toc; 
    end
    
    disp('d');

    if TRMstep
        tic
        [p0, sigma_p0, next_error0, valid_p0] = getP0(W1,W2,bias1,bias2,g1s,g1_1s,g2_1s,g2_2s,g1_2s,dg1s,dg2s,input_set,output_set,lambda,g,gamma,WS,indices,b_w,sub_tol,sub_maxit);
        t2 = toc;
        
        tic
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
        
        if ((1/m)*sqrt(g.'*g) < tol)
            break;
        end
        t3 = toc;
    else
        tic
        [W1,W2,bias1,bias2] = addP(p1,n0,n1,n2,W1,W2,bias1,bias2);
        t2 = toc;
        % some update schedule
        tic
        gamma = lr / (1 + decay*k);
        % some stopping condition
        if gamma < 10^(-2)
            break;
        end
        t3 = toc;
       
    end
        
    t = subset_m_time + g_time + subset_w_time + p1_time + t2 + t3;
    rolling_t = rolling_t + t;
    if mod(k,50) == 0;
        disp(k);
    end
    
    if MS == false
        full_error = last_error;
        g_total = g_full;
    end
    
    if tofile
        fprintf(file,'%d, %d, %d, %d, %d, %d, %d, %d, %f, %e, %e, %f, %e, %d, %f, %e \n', rolling_t, t, subset_m_time, g_time, subset_w_time,p1_time,t2,t3, full_error, (1/m)*sqrt(g_total.'*g_total), rho, sigma, gamma, step, last_error, (1/m)*sqrt(g_full.'*g_full));
    else
        disp(full_error);
    end
end