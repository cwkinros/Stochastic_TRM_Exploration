function [] = testParams(dataset,sgd_lr_range,lr_range,b_w_range,b_m_mini_range,b_m_big_range,maxiter,revision)

if revision
    params = readtable(strcat('results/',dataset,'params_results.txt'));
    sgd_lr = params.sgdlr;
    lr = params.lr;
    b_w = params.bw;
    b_m_mini = params.bmmini;
    b_m_big = params.bmbig;    
end

file = fopen(strcat('results/',dataset,'params_results.txt'),'w');
fprintf(file,'sgdlr, lr, bw, bmmini, bmbig \n');

if sgd_lr_range ~= false
    min = inf;
    i = sgd_lr_range(1);
    while true
        error = run_test(dataset,10^i,0,0,0,0,'SGD',true,'sgd_lr',i,maxiter);
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
        error = run_test(dataset,0,10^i,0,0,0,'STRM',true,'lr',i,maxiter);
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
        error = run_test(dataset,0,lr,0,i,0,'MBTRM',true,'b_m_mini',i,maxiter);
        if error < min
            min = error;
            b_m_mini = i;
        else
            break;
        end
        i = i + b_m_mini_range(2);
    end
end

if b_m_big_range ~= false
    min = inf;
    i = b_m_big_range(1);
    while true
        % error = inf if b_m_mini > m
        error = run_test(dataset,0,0,0,0,i,'BTRM',true,'b_m_big',i,maxiter);
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
        error = run_test(dataset,0,lr,i,0,b_m_big,'TRM_WS',true,'b_w',i,maxiter);
        if error < min
            min = error;
            b_w = i;
        else
            break;
        end
        i = i + b_w_range(2);
    end
end
fprintf(file,'%d, %d, %d, %d, %d',sgd_lr,lr,b_w,b_m_mini,b_m_big);

