function [] = testDERM()

[inputs,outputs] = getDermData();

[n0,~] = size(inputs);


n1 = 10;
[n2,~] = size(outputs);


% + 1 for the bias
[W1,W2,bias1,bias2] = getWeightsFromFile('Derm_initialWeights.txt',n0,n1,n2);



maxiter = 10000000;

lr = 1.0;
b_w = 35;
b_m_mini = 12;
b_m_big = 340;

% for STRM - lr = 1.5

if SGD
%-------------------- test SGD -----------------------------------------
    disp('test SGD');
    WS = false;
    MS = 1;
    TRMstep = false;
    tofile = true;
    GD = true;
    file = fopen('results/Derm_SGD.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2,tofile, file);
end

if MBGD
%----------------- test MBGD -----------------------------------------
    disp('test MBGD');
    WS = false;
    MS = 2;
    TRMstep = false;
    tofile = true;
    file = fopen('results/Derm_MBGD.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, b_m_mini*lr);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);
end

%---------------------- no longer GD-----------------------------------
GD = false;

%----------------------------------------------------------------------
if TRM_WS
    %---------------------- test TRM_WS ------------------------------------
    disp('test TRM_WS');
    WS = true;
    MS = 0;
    TRMstep = true;
    file = fopen('results/Derm_TRM_WS.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);
end

if STRM
    %---------------------- test STRM ------------------------------------
    disp('test STRM');
    WS = false;
    MS = 1;
    TRMstep = false;
    file = fopen('results/Derm_STRM.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);
end

if MBTRM
    %---------------------- test MBTRM ------------------------------------
    disp('test MBTRM');
    WS = false;
    MS = 2;
    TRMstep = false;
    file = fopen('results/Derm_MBTRM.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, b_m_mini*lr);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);
end

if STRM_WS
    %---------------------- test STRM_WS ------------------------------------
    disp('test STRM_WS');
    WS = true;
    MS = 1;
    TRMstep = false;
    file = fopen('results/Derm_STRM_WS.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);
end

if MBTRM_WS
    %---------------------- test MBTRM_WS ------------------------------------
    disp('test MBTRM_WS');
    WS = true;
    MS = 2;
    TRMstep = false;
    file = fopen('results/Derm_MBTRM_WS.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, b_m_mini*lr);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);
end

if BTRM
    %---------------------- test BTRM ------------------------------------
    disp('test BTRM');
    WS = false;
    MS = 2;
    TRMstep = true;
    file = fopen('results/Derm_BTRM.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);    
end

if BTRM_WS
    %---------------------- test BTRM_WS ------------------------------------
    disp('test BTRM_WS');
    WS = true;
    MS = 2;
    TRMstep = true;
    file = fopen('results/Derm_BTRM_WS.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
    print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file); 
end

