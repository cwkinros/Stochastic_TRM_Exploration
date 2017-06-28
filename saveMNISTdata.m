function [] = saveMNISTdata()

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

st = '';
for i = 1:99
    st = strcat(st,'%f ,');
end
st = strcat(st,'%f \n');

file = fopen('mnist_inputs.txt','w');
fprintf(file,st,inputs);
fclose(file);

file = fopen('mnist_outputs.txt','w');
fprintf(file,'%f, %f, %f, %f, %f, %f, %f, %f, %f, %f \n', outputs);
fclose(file);


% make sure to include a bias
