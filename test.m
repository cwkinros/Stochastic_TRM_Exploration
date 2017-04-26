function [] = run_test(dataset,sgd_lr,lr,b_w,b_m_mini,b_m_big,tests)

if strcmp('MNIST',dataset)
    [inputs,outputs] = getMNISTData();
else if strcmp('Derm',dataset)
        [inputs,outputs] = getDermData();
    else if strcmp('IRIS',dataset)
            [inputs,outputs] = getIrisData();
        else if strcmp('Nurs',dataset)
                [inputs,outputs] = getNurseryParams();
            else
                disp(strcat('The dataset: ', dataset, ' is not available!'));
            end
        end
    end
end

            
[n0,~] = size(inputs);
n1 = 10;
[n2,~] = size(outputs);


weights_filename = strcat(dataset,'_initialWeights.txt');
[W1,W2,bias1,bias2] = getWeightsFromFile(weights_filename,n0,n1,n2);

maxiter = 10;

[~,len] = size(tests);
i = 1;
while i <= len
    start = i;
    while i <= len && tests(i) ~= ' '
        i = i + 1;
    end
    test = tests(start:i-1);
    i = i + 1;
    
    [WS,MS,TRMstep,GD,gamma] = getParams(test,lr,sgd_lr,b_m_mini);
    file = fopen(strcat('results/',dataset,'_',test,'.txt'),'w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, gamma);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2,tofile, file);
    w_final = M1M2_to_m(final_W1,final_W2,final_bias1,final_bias2);
    final_weights = fopen(strcat('results/',dataset,'_finalWeights_',test,'.txt'),'w');
    fprintf(final_weights,'%d \n',w_final);
end
    



function [WS,MS,TRMstep,GD,gamma] = getParams(test,lr,sgd_lr,b_m_mini)
WS = false;
MS = 0;
TRMstep = false;
GD = false;
gamma = 1;

if strcmp(test,'BTRMWS')
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
        else if strcmp(test,'MBSGD')
                MS = 2;
                GD = true;
                gamma = b_m_mini*sgd_lr;
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
                                    gamma = lr*b_m_mini;
                                else if strcmp(test,'MBTRM_WS')
                                        WS = true;
                                        MS = 2;
                                        gamma = lr*b_m_mini;
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



                        
