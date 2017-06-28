function [error, rolling_t] = run_test(dataset,sgd_lr,sgd_lr_mb,lr,lr_mb,b_w,b_m_mini,b_m_mini_MBGD,b_m_big,tests,testparams,var,val,maxiter,n1,sub_maxit_val, altern, time_lim)

if strcmp('MNIST',dataset)
    [inputs,outputs] = getMNISTdata();
else if strcmp('Derm',dataset)
        [inputs,outputs] = getDermData();
    else if strcmp('IRIS',dataset)
            [inputs,outputs] = getIrisData();
        else if strcmp('Nurs',dataset)
                [inputs,outputs] = getNurseryParams();
            else if strcmp('Habe',dataset)
                    [inputs,outputs] = getHabermanData();
                else if strcmp('XOR',dataset)
                        [inputs,outputs] = getXORdata();
                    else
                        disp(strcat('The dataset: ', dataset, ' is not available!'));
                    end
                end
            end
        end
    end
end

wbias = true;     
[n0,m] = size(inputs);
if n1 == 0
    n1 = 10;
end
[n2,~] = size(outputs);
if wbias
    n = n2*(n1 + 1) + n1*(n0 + 1);
else
    n = n2*n1 + n1*n0;
end

if b_m_mini > m || b_m_big > m || b_w > n
    error = inf;
    return;
end

weights_filename = strcat(dataset,'_initialWeights.txt');
[W1,W2,bias1,bias2] = getWeightsFromFile(weights_filename,n0,n1,n2);
if wbias == false
    bias1 = NaN;
    bias2 = NaN;
end

tofile = true;

[~,len] = size(tests);
i = 1;
while i <= len
    start = i;
    while i <= len && tests(i) ~= ' '
        i = i + 1;
    end
    test = tests(start:i-1);
    i = i + 1;
    
    [WS,MS,TRMstep,GD,gamma,b_m_mini_general] = getParams(test,lr,sgd_lr,lr_mb,sgd_lr_mb,b_m_mini,b_m_mini_MBGD);
    if tofile
        if testparams
            file = fopen(strcat('results/test',var,'_',dataset,num2str(val),'.txt'),'w');
        else
            file = fopen(strcat('results/',dataset,'_',test,'.txt'),'w');
        end
    else 
        file = 0;
    end
    [final_W1, final_W2, final_bias1, final_bias2, error, rolling_t] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini_general, b_m_big, gamma, sub_maxit_val,[],[],altern,time_lim);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2,tofile, file);
    w_final = M1M2_to_m(final_W1,final_W2,final_bias1,final_bias2);
    if testparams
       final_weights = fopen(strcat('results/test',var,'_',dataset,num2str(val),'_finalw.txt'),'w');
    else
       final_weights = fopen(strcat('results/',dataset,'_',test,'_finalw.txt'),'w');
    end
    fprintf(final_weights,'%d \n',w_final);
end
    



function [WS,MS,TRMstep,GD,gamma,b_m_mini_general] = getParams(test,lr,sgd_lr,lr_mb,sgd_lr_mb,b_m_mini,b_m_mini_MBGD)
WS = false;
MS = 0;
TRMstep = false;
GD = false;
gamma = 1;
b_m_mini_general = b_m_mini;
if strcmp(test,'BTRM_WS')
    WS = true;
    MS = 2;
    TRMstep = true;
else if strcmp(test,'BTRM')
        MS = 2;
        TRMstep = true;
    else if strcmp(test,'SGD')
            MS = 1;
            GD = true;
            gamma = sgd_lr;
        else if strcmp(test,'MBGD')
                MS = 2;
                GD = true;
                b_m_mini_general = b_m_mini_MBGD;
                gamma = sgd_lr_mb;
            else if strcmp(test,'TRM')
                    TRMstep = true;
                else if strcmp(test,'TRM_WS')
                        WS = true;
                        TRMstep = true;
                    else if strcmp(test,'STRM')
                            MS = 1;
                            gamma = lr;
                        else if strcmp(test,'STRM_WS')
                                WS = true;
                                MS = 1;
                                gamma = lr;
                            else if strcmp(test,'MBTRM')
                                    MS = 2;
                                    gamma = lr_mb;
                                else if strcmp(test,'MBTRM_WS')
                                        WS = true;
                                        MS = 2;
                                        gamma = lr_mb;
                                    else if strcmp(test,'TRM_MBGD')
                                            GD = true;
                                            TRMstep = true;
                                            MS = 2;
                                            b_m_mini_general = b_m_mini_MBGD;
                                            gamma = sgd_lr_mb;
                                        else if strcmp(test,'STRM_NC')
                                                MS = 1;
                                                gamma = lr;       
                                        
                                            else if strcmp(test,'GD')
                                                    GD = true;
                                                    gamma = sgd_lr;
                                                else
                                                disp(strcat('Method: ',test,' is not covered'));
                                            
                                                end
                                            end
                                        
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end



                        
