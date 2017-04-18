function [W1, W2, bias1, bias2] = train_TRM_united(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1_given, maxiter, tofile, file)

[~,n0] = size(W1);
[n2,n1] = size(W2);
n = n2*(n1 + 1) + n1*(n0 + 1);

% b_w should be about 24% of  n from test
b_w = round(n*0.24,0);

[~,m] = size(inputs);

b_m_mini = 10;

% b_m is best at about 50%
b_m_big = round(0.5*m,0);

% tested using sample test 
% lr determined to be largest productive lr when decay = 0
% decay determined for max iter ~10^7
lr = 1.7;
decay = 10^(-4);

[W1, W2, bias1, bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr, decay);