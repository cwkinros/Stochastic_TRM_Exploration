function [] = testNurseryParams()

[inputs,outputs] = getNurseryParams();

[n0,~] = size(inputs);
n1 = 10;
n2 = 5;

n = n2*(n1 + 1) + n1*(n0 + 1);

W1 = rand(n1, n0) - 0.5;
W1 = W1/sum(sum(abs(W1)));
W2 = rand(n2, n1) - 0.5;
W2 = W2/sum(sum(abs(W2)));

bias1 = rand(n1,1) - 0.5;
bias2 = rand(n2,1) - 0.5;

% test for b_w

maxiter = 100000;
tofile = false;
b_m_mini = 0;
b_m_big = 1000;
lr = 1;
GD = false;
WS = true;
MS = 2;
TRMstep = true;
for b_w = 5:10:50
    file = fopen(strcat('results/testb_w_Nurs',int2str(b_w),'.txt'),'w');
    [~,~,~,~] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
end

WS = false;
MS = 1;
TRMstep = false;
for i = -5:2
    lr = 10^i;
    file = fopen(strcat('results/test_LR_Nurs',int2str(i),'.txt'),'w');
    [~,~,~,~] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);
end

WS = false;
MS = 2;
TRMstep = false;
lr = 1;
for b_m_mini = 3:3:12
    file = fopen(strcat('results/testb_m_mini_Nurs',int2str(b_m_mini),'.txt'),'w');
    [~,~,~,~] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);    
end

WS = false;
MS = 2;
TRMstep = true;
lr = 1;
for b_m_big = 500:300:2000
    file = fopen(strcat('results/testb_m_big_Nurs',int2str(b_m_big),'.txt'),'w');
    [~,~,~,~] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);    
end

GD = true;
WS = false;
MS = 1;
TRMstep = false;
for i = -5:2
    lr = 10^i;
    file = fopen(strcat('results/testGDlr_Nurs',int2str(lr*10),'.txt'),'w');
    [~,~,~,~] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr);    
end