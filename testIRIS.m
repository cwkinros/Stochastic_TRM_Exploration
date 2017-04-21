function [] = testIRIS()

[inputs,outputs] = getIrisData();

[n0,~] = size(inputs);
n1 = 10;
[n2,~] = size(outputs);

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
file_weights = fopen('results/IRIS_initialWeights.txt','w');
fprintf(file_weights,'%f \n',w);



maxiter = 500000;

b_m_mini = 9;
b_m_big = 150;
b_w = 15;
sgd_lr = 1;
lr = 0.01; % for STRM
%-------------------- test SGD -----------------------------------------
disp('test SGD');
WS = false;
MS = 1;
TRMstep = false;
tofile = false;
GD = true;
file = fopen('results/IRIS_SGD.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, sgd_lr);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, tofile, file);

%----------------- test MBGD -----------------------------------------
disp('test MBGD');
WS = false;
MS = 2;
TRMstep = false;
tofile = false;
GD = true;
file = fopen('results/IRIS_MBGD.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, sgd_lr*b_m_mini);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, tofile, file);

%---------------------- no longer GD-----------------------------------
GD = false;
%----------------------------------------------------------------------

%---------------------- test TRM_WS ------------------------------------
disp('test TRM_WS');
WS = true;
MS = 0;
TRMstep = true;
tofile = true;
file = fopen('results/IRIS_TRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test STRM ------------------------------------
disp('test STRM');
WS = false;
MS = 1;
TRMstep = false;
tofile = true;
file = fopen('results/IRIS_STRM.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);



%---------------------- test STRM_WS ------------------------------------
disp('test STRM_WS');
WS = true;
MS = 1;
TRMstep = false;
tofile = true;
file = fopen('results/IRIS_STRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test MBTRM_WS ------------------------------------
GD = false;
disp('test MBTRM_WS');
WS = true;
MS = 2;
TRMstep = false;
tofile = true;
file = fopen('results/IRIS_MBTRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, b_m_mini*lr);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);


%---------------------- test MBTRM ------------------------------------
disp('test MBTRM');
WS = false;
MS = 2;
TRMstep = false;
tofile = true;
file = fopen('results/IRIS_MBTRM.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr*b_m_mini);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test TRM ------------------------------------
disp('test TRM');
WS = false;
MS = 0;
TRMstep = true;
tofile = true;
file = fopen('results/IRIS_TRM.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2, true, file);

