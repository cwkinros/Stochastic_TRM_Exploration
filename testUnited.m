function [] = testUnited()

images = loadMNISTImages('train-images.idx3-ubyte');
 
% labels: 60000 by 1 matrix (vector) of labels 
labels = loadMNISTLabels('train-labels.idx1-ubyte');
 
% number of inputs k0
%k0 = 784;
start = 300;
 
k0 = 25;
 

m = 60000;

inputs = images(start+1:start+k0,1:m);
labels = labels(1:m);

outputs = zeros(10,m);
for i = 1:m
    if (labels(i) == 0)
        labels(i) = 10;
    end
    outputs(labels(i),i) = 1;
end

% make sure to include a bias

n0 = k0; 
n1 = 10;
n2 = 10;

n = n0*n1 + n1*n2 + n1 + n2;


% + 1 for the bias
W1 = rand(n1, n0) - 0.5;
W1 = W1/sum(sum(abs(W1)));
W2 = rand(n2, n1) - 0.5;
W2 = W2/sum(sum(abs(W2)));

bias1 = rand(n1,1) - 0.5;
bias2 = rand(n2,1) - 0.5;

maxiter = 100000000;

%---------------------- test full TRM ------------------------------------
disp('test full TRM');
WS = false;
MS = 0;
TRMstep = true;
tofile = true;
file = fopen('results/TRM.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test TRM_WS ------------------------------------
disp('test TRM_WS');
WS = true;
MS = 0;
TRMstep = true;
tofile = true;
file = fopen('results/TRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test STRM ------------------------------------
disp('test STRM');
WS = false;
MS = 1;
TRMstep = false;
tofile = true;
file = fopen('results/STRM.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test MBTRM ------------------------------------
disp('test MBTRM');
WS = false;
MS = 2;
TRMstep = false;
tofile = true;
file = fopen('results/MBTRM.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test STRM_WS ------------------------------------
disp('test STRM_WS');
WS = true;
MS = 1;
TRMstep = false;
tofile = true;
file = fopen('results/STRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test MBTRM_WS ------------------------------------
disp('test MBTRM_WS');
WS = true;
MS = 2;
TRMstep = false;
tofile = true;
file = fopen('results/MBTRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);

%---------------------- test BTRM ------------------------------------
disp('test BTRM');
WS = false;
MS = 2;
TRMstep = true;
tofile = true;
file = fopen('results/BTRM.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);    

%---------------------- test BTRM_WS ------------------------------------
disp('test BTRM_WS');
WS = true;
MS = 2;
TRMstep = true;
tofile = true;
file = fopen('results/BTRM_WS.txt','w');
[final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united(WS,MS,TRMstep,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file);
print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);  