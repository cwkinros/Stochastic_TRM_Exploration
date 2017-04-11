function [] = testP0_optim()

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
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


n0 = k0; 
n1 = 10;
n2 = 10;

n = n0*n1 + n1*n2 + n1 + n2;


% + 1 for the bias
W1 = rand(n1, n0) - 0.5;
W1 = W1/sum(sum(abs(W1)));
W2 = rand(n2, n1) - 0.5;
W2 = W2/sum(sum(abs(W2)));

bias1 = (rand(n1,1) - 0.5);
bias1 = bias1/sum(abs(bias1));
bias2 = (rand(n2,1) - 0.5);
bias2 = bias2/sum(abs(bias2));

file1 = fopen('results\test_p0_method1.txt','w');
file2 = fopen('results\test_p0_method2.txt','w');
file3 = fopen('results\test_p0_method3.txt','w');
file4 = fopen('results\test_p0_method4.txt','w');

[~,~,~,~] = train_TRM(inputs,outputs, W1, W2, bias1, bias2, n1, 1000, 1, file1); 
[~,~,~,~] = train_TRM(inputs,outputs, W1, W2, bias1, bias2, n1, 1000, 2, file2); 
[~,~,~,~] = train_TRM(inputs,outputs, W1, W2, bias1, bias2, n1, 1000, 3, file3); 
[~,~,~,~] = train_TRM(inputs,outputs, W1, W2, bias1, bias2, n1, 1000, 4, file4); 


    