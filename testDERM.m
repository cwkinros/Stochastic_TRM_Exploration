function [] = testDERM()

[inputs,outputs] = getDermData();

[n0,~] = size(inputs);


n1 = 10;
n2 = 6;


% + 1 for the bias
W1 = rand(n1, n0) - 0.5;
W1 = W1/sum(sum(abs(W1)));
W2 = rand(n2, n1) - 0.5;
W2 = W2/sum(sum(abs(W2)));

bias1 = rand(n1,1) - 0.5;
bias2 = rand(n2,1) - 0.5;

w = M1M2_to_m(W1,W2,bias1,bias2);
file_weights = fopen('results/Derm_initialWeights.txt','w');
fprintf(file_weights,'%f \n',w);



maxiter = 100000;

lr = 1.0;
b_w = 5;
b_m_mini = 6;
b_m_big = 180;
% for STRM - lr = 1.5

%-------------------- test SGD -----------------------------------------
disp('test SGD');
WS = false;
MS = 1;
TRMstep = false;
tofile = true;
GD = true;
decay = 10^(-3);
file = fopen('results/Derm_SGD.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr, decay);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);

%----------------- test MBGD -----------------------------------------
disp('test MBGD');
WS = false;
MS = 2;
TRMstep = false;
tofile = true;
GD = true;
file = fopen('results/Derm_MBGD.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);


%---------------------- no longer GD-----------------------------------
GD = false;
lr = 1.5;
%----------------------------------------------------------------------

%---------------------- test TRM_WS ------------------------------------
disp('test TRM_WS');
WS = true;
MS = 0;
TRMstep = true;
tofile = true;
file = fopen('results/Derm_TRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test STRM ------------------------------------
disp('test STRM');
WS = false;
MS = 1;
TRMstep = false;
tofile = true;
file = fopen('results/Derm_STRM.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test MBTRM ------------------------------------
disp('test MBTRM');
WS = false;
MS = 2;
TRMstep = false;
tofile = true;
file = fopen('results/Derm_MBTRM.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test STRM_WS ------------------------------------
disp('test STRM_WS');
WS = true;
MS = 1;
TRMstep = false;
tofile = true;
file = fopen('results/Derm_STRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test MBTRM_WS ------------------------------------
disp('test MBTRM_WS');
WS = true;
MS = 2;
TRMstep = false;
tofile = true;
file = fopen('results/Derm_MBTRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test BTRM ------------------------------------
disp('test BTRM');
WS = false;
MS = 2;
TRMstep = true;
tofile = true;
file = fopen('results/Derm_BTRM.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);    

%---------------------- test BTRM_WS ------------------------------------
disp('test BTRM_WS');
WS = true;
MS = 2;
TRMstep = true;
tofile = true;
file = fopen('results/Derm_BTRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file); 


