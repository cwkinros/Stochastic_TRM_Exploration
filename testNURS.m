function [] = testNURS()

[inputs,outputs] = getNurseryParams();

[n0,~] = size(inputs);
n1 = 10;
n2 = 5;

% make sure to include a bias


n = n0*n1 + n1*n2 + n1 + n2;


% + 1 for the bias
W1 = rand(n1, n0) - 0.5;
W1 = W1/sum(sum(abs(W1)));
W2 = rand(n2, n1) - 0.5;
W2 = W2/sum(sum(abs(W2)));

bias1 = rand(n1,1) - 0.5;
bias2 = rand(n2,1) - 0.5;

w = M1M2_to_m(W1,W2,bias1,bias2);
file_weights = fopen('results/NURS_initialWeights.txt','w');
fprintf(file_weights,'%f \n',w);



maxiter = 10000000;
if false
    %-------------------- test SGD -----------------------------------------
    disp('test SGD');
    WS = false;
    MS = 1;
    TRMstep = false;
    tofile = true;
    GD = true;
    file = fopen('results/NURS_SGD.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
    print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);

    %----------------- test MBGD -----------------------------------------
    disp('test MBGD');
    WS = false;
    MS = 2;
    TRMstep = false;
    tofile = true;
    GD = true;
    file = fopen('results/NURS_MBGD.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
    print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);


    %---------------------- no longer GD-----------------------------------
    GD = false;
    %----------------------------------------------------------------------

    %---------------------- test TRM_WS ------------------------------------
    disp('test TRM_WS');
    WS = true;
    MS = 0;
    TRMstep = true;
    tofile = true;
    file = fopen('results/NURS_TRM_WS.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
    print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);

    %---------------------- test STRM ------------------------------------
    disp('test STRM');
    WS = false;
    MS = 1;
    TRMstep = false;
    tofile = true;
    file = fopen('results/NURS_STRM.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
    print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);



    %---------------------- test STRM_WS ------------------------------------
    disp('test STRM_WS');
    WS = true;
    MS = 1;
    TRMstep = false;
    tofile = true;
    file = fopen('results/NURS_STRM_WS.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
    print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);
    
    %---------------------- test MBTRM_WS ------------------------------------
    GD = false;
    disp('test MBTRM_WS');
    WS = true;
    MS = 2;
    TRMstep = false;
    tofile = true;
    file = fopen('results/NURS_MBTRM_WS.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
    print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);
    
    %---------------------- test BTRM_WS ------------------------------------
    disp('test BTRM_WS');
    WS = true;
    MS = 2;
    TRMstep = true;
    tofile = true;
    file = fopen('results/NURS_BTRM_WS.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
    print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file); 

end

if true
    %---------------------- test BTRM ------------------------------------
    GD = false;
    disp('test BTRM');
    WS = false;
    MS = 2;
    TRMstep = true;
    tofile = true;
    file = fopen('results/NURS_BTRM.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
    print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);    

    %---------------------- test MBTRM ------------------------------------
    disp('test MBTRM');
    WS = false;
    MS = 2;
    TRMstep = false;
    tofile = true;
    file = fopen('results/NURS_MBTRM.txt','w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
    print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);
end
