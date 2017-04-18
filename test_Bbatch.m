function [] = test_Bbatch()

%using a drop based schedule as described below
% http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/

images = loadMNISTImages('train-images.idx3-ubyte');
 
% labels: 60000 by 1 matrix (vector) of labels 
labels = loadMNISTLabels('train-labels.idx1-ubyte');
 
% number of inputs k0
%k0 = 784;
start = 300;
 
k0 = 10;
 

m = 100;

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

maxiter = 10000;

%---------------------- test BTRM ------------------------------------
disp('test BTRM');
WS = false;
MS = 2;
TRMstep = true;
lr = 0;
decay = 0;

tofile = true;
b_w = 0;
b_m_mini = 0;
% note m = 350
for b_m_big = 20:10:60
    file = fopen(strcat('results/BTRM',int2str(b_m_big),'.txt'),'w');
    [final_W1, final_W2, final_bias1, final_bias2] = train_TRM_united_w_param_control(WS,MS,TRMstep,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, lr, decay);
    print_accuracy(inputs,labels, final_W1, final_W2, final_bias1, final_bias2, true, file);
end