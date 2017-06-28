function [] = testParams(dataset,sgd_lr_range,lr_range,b_w_range,b_m_mini_range,b_m_big_range,sgd_lr_mb_range, lr_mb_range, b_m_mini_MBGD_range, maxiter,revision,n1)


[~,time_lim] = run_test(dataset,0,0,0,0,0,0,0,0,'TRM',false,false,false,10,10,0,true,0);

if revision
    params = readtable(strcat('results/',dataset,'params_results.txt'));
    sgd_lr = params.sgdlr;
    lr = params.lr;
    b_w = params.bw;
    b_m_mini = params.bmmini;
    b_m_big = params.bmbig;    
end


file = fopen(strcat('results/',dataset,'params_results.txt'),'w');
fprintf(file,'sgdlr, sgdlrmb, lr, lrmb, bw, bmmini, bmbig, bmmini_MBGD\n');

if sgd_lr_range ~= false
    min = inf;
    i = sgd_lr_range(1);
    while true
        [error,~] = run_test(dataset,10^i,0,0,0,0,'SGD',true,'sgd_lr',i,maxiter,n1,0,true,time_lim);
        if error < min
            min = error;
            sgd_lr = 10^i;
        else
            break;
        end
        i = i + sgd_lr_range(2);
    end
end



if lr_range ~= false
    min = inf;
    i = lr_range(1);
    while true
        [error,~] = run_test(dataset,0,10^i,0,0,0,'STRM',true,'lr',i,maxiter,n1,0,true,time_lim);
        if error < min
            min = error;
            lr = 10^i;
        else
            break
        end
        i = i + lr_range(2);
    end
end

if b_m_mini_range ~= false
    min = inf;
    i = b_m_mini_range(1);
    while true
        % error = inf if b_m_mini > m
        [error,~] = run_test(dataset,0,lr,0,i,0,'MBTRM',true,'b_m_mini',i,maxiter,n1,0,true,time_lim);
        if error < min
            min = error;
            b_m_mini = i;
        else
            break;
        end
        i = i + b_m_mini_range(2);
    end
end

if b_m_mini_MBGD_range ~= false
    min = inf;
    i = b_m_mini_MBGD_range(1);
    while true
        % error = inf if b_m_mini > m
        [error,~] = run_test(dataset,sgd_lr,0,0,i,0,'MBGD',true,'b_m_mini_MBGD',i,maxiter,n1,0,true,time_lim);
        if error < min
            min = error;
            b_m_mini_MBGD = i;
        else
            break;
        end
        i = i + b_m_mini_MBGD_range(2);
    end
end

if sgd_lr_mb_range ~= false
    min = inf;
    i = sgd_lr_mb_range(1);
    while true
        [error,~] = run_test(dataset,10^i,0,0,b_m_mini_MBGD,0,'MBGD',true,'sgd_lr_mb',i,maxiter,n1,0,true,time_lim);
        if error < min
            min = error;
            sgd_lr_mb = 10^i;
        else
            break;
        end
        i = i + sgd_lr_mb_range(2);
    end
end

if lr_mb_range ~= false
    min = inf;
    i = lr_mb_range(1);
    while true
        [error,~] = run_test(dataset,0,10^i,0,b_m_mini,0,'MBTRM',true,'lr',i,maxiter,n1,0,true,time_lim);
        if error < min
            min = error;
            lr_mb = 10^i;
        else
            break
        end
        i = i + lr_mb_range(2);
    end
end


if b_m_big_range ~= false
    min = inf;
    i = b_m_big_range(1);
    while true
        % error = inf if b_m_mini > m
        [error,~] = run_test(dataset,0,0,0,0,i,'BTRM',true,'b_m_big',i,maxiter,n1,0,true,time_lim);
        if error < min
            min = error;
            b_m_big = i;
        else
            break;
        end
        i = i + b_m_big_range(2);
    end
end

if b_w_range ~= false
    min = inf;
    i = b_w_range(1);
    while true
        % error = inf if b_m_mini > m
        [error,~] = run_test(dataset,0,lr,i,0,b_m_big,'TRM_WS',true,'b_w',i,maxiter,n1,0,true,time_lim);
        if error < min
            min = error;
            b_w = i;
        else
            break;
        end
        i = i + b_w_range(2);
    end
end

fprintf(file,'%d, %d, %d, %d, %d, %d, %d, %d',sgd_lr,sgd_lr_mb,lr,lr_mb,b_w,b_m_mini,b_m_big,b_m_mini_MBGD);

