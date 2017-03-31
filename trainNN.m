function [gammas,rhos,gmag, errorTRM1] = trainNN(W1,W2,bias1,bias2)

% two layers
% we have w = {W1, W2}, g = {GradW1, GradW2}

images = loadMNISTImages('train-images.idx3-ubyte');
 
% labels: 60000 by 1 matrix (vector) of labels 
labels = loadMNISTLabels('train-labels.idx1-ubyte');
 
% number of inputs k0
%k0 = 784;
start = 300;
 
k0 = 20;
 

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
W1 = rand(n1, n0);
W1 = W1/sum(sum(abs(W1)));
W2 = rand(n2, n1);
W2 = W2/sum(sum(abs(W2)));

bias1 = rand(n1,1);
bias2 = rand(n2,1);


%[W1,W2,bias1,bias2,~] =  train_BTRM_WS(inputs,outputs, W1, W2, bias1, bias2, n1,500,30);
[W1,W2,bias1,bias2,~] =  train_MBTRM_WS(inputs,outputs, W1, W2, bias1, bias2, n1,10,10);
%[W1,W2,bias1,bias2,~] =  train_STRM_WS(inputs,outputs, W1, W2, bias1, bias2, n1, 10);
%[W1,W2,bias1,bias2,~] =  train_TRM_WS(inputs,outputs, W1, W2, bias1, bias2, n1, 3);
%[W1,W2,bias1,bias2,~] =  train_GD(inputs,outputs, W1, W2, bias1, bias2);
%[W1,W2,bias1,bias2,errorTRM1, gammas, rhos, gmag] = train_TRM(inputs,outputs, W1, W2, bias1, bias2, n1, 100); 
%[~,~,~,~,errorTRM2] = train_TRM(inputs,outputs, W1, W2, bias1, bias2, n1, 10000); 
%[W1,W2,bias1,bias2,errorSTRM] = train_STRM(inputs,outputs, W1, W2, bias1, bias2, n1);
%[~,~,~,~,errorSTRM_RMI] = train_STRM_RMI(inputs,outputs, W1, W2, bias1, bias2, n1);
%[W1,W2,bias1,bias2,errorMBTRM] = train_MBTRM(inputs,outputs, W1, W2, bias1, bias2, n1, 10);
%[W1,W2,bias1,bias2,errorBTRM] = train_BTRM(inputs,outputs, W1, W2, bias1, bias2, n1, 200);
print_accuracy(inputs,labels, W1, W2, bias1, bias2);

if false
    plot(errorTRM1);
    hold on;
    plot(errorTRM2);
    legend('TRM1','TRM2');
    return;
    plot(errorSTRM);
    hold on;
    plot(errorMBTRM);
    hold on;
    plot(errorBTRM);
    hold on;
    plot(errorSTRM_RMI);
    legend('TRM','STRM','MBTRM','BTRM','STRM_RMI');
    disp('results:');
    %disp(bias1);
    %disp(bias2);
    %disp(W1);
    %disp(W1);
end
    
    