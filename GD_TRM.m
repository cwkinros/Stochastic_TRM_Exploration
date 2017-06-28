% here we commpare GD with TRM for a very simple test

w1 = 0;
w2 = 0;
n0 = 1;
n1 = 1;
n2 = 1;
inputs = 5;
outputs = 0.8;
bias1 = 0;
bias2 = 0;
n = 4;
lambda =0;
m=1;
maxiter = 200000;
tofile = true;
file = fopen('TRM_convergence.txt','w');

WS = false;
MS = 0;
TRMstep = true;
GD = false;
[final_W1, final_W2, final_bias1, final_bias2, full_error] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, w1, w2, bias1, bias2, n1, maxiter, tofile, file, 0, 0, 0, 1, 0);
fclose(file);
file = fopen('GD_convergence.txt','w');
GD = true;
TRMstep = false;
[final_W1, final_W2, final_bias1, final_bias2, full_error] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, w1, w2, bias1, bias2, n1, maxiter, tofile, file, 0, 0, 0, 1, 0);
