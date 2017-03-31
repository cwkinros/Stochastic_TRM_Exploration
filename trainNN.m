function [] = trainNN()

% two layers
% we have w = {W1, W2}, g = {GradW1, GradW2}

images = loadMNISTImages('train-images.idx3-ubyte');
 
% labels: 60000 by 1 matrix (vector) of labels 
labels = loadMNISTLabels('train-labels.idx1-ubyte');
 
% number of inputs k0
%k0 = 784;
start = 300;
 
k0 = 20;
 
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
W1 = rand(n1, n0);
W1 = W1/sum(sum(abs(W1)));
W2 = rand(n2, n1);
W2 = W2/sum(sum(abs(W2)));

bias1 = rand(n1,1);
bias2 = rand(n2,1);






%[~,~,~,~,errorTRM] = train_TRM(inputs,outputs, W1, W2, bias1, bias2, n1); 
%[~,~,~,~,errorSTRM] = train_STRM(inputs,outputs, W1, W2, bias1, bias2, n1);
[~,~,~,~,errorSTRM_RMI] = train_STRM_RMI(inputs,outputs, W1, W2, bias1, bias2, n1);
%[~,~,~,~,errorMBTRM] = train_MBTRM(inputs,outputs, W1, W2, bias1, bias2, n1, 10);
%[~,~,~,~,errorBTRM] = train_BTRM(inputs,outputs, W1, W2, bias1, bias2, n1, 200);
print_accuracy(inputs,labels, W1, W2, bias1, bias2);
plot(errorSTRM_RMI);
plot(errorTRM);
hold on;
plot(errorSTRM);
hold on;
plot(errorMBTRM);
hold on;
plot(errorBTRM);
disp('results:');
%disp(bias1);
%disp(bias2);
%disp(W1);
%disp(W1);
    
    