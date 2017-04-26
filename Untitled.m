function [inputs,outputs,n0,n1,n2] = getMNISTdata()

images = loadMNISTImages('train-images.idx3-ubyte');
 
% labels: 60000 by 1 matrix (vector) of labels 
labels = loadMNISTLabels('train-labels.idx1-ubyte');

m = 60000;
disp('down sampling');
inputs = downsample_mnist(images(:,1:m));
disp('finished down sampling');
labels = labels(1:m);

outputs = zeros(10,m);
for i = 1:m
    if (labels(i) == 0)
        labels(i) = 10;
    end
    outputs(labels(i),i) = 1;
end

% make sure to include a bias

n0 = 100; 
n1 = 10;
n2 = 10;