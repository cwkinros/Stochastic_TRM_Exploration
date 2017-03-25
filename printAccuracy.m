function [] = print_accuracy(inputs,labels,W1,W2)

outputs = sigmoid(W2*sigmoid(W1*inputs));
[r,c] = size(outputs);
sum = 0;
for i = 1:c
    max = -1;
    max_idx = -1;
    for j = 1:r
        if outputs(j,i) > max
            max = outputs(j,i);
            max_idx = j;
        end
    end
    if labels(i) == max_idx || (labels(i) == 0 && max_idx == 10):
        sum = sum + 1;
    end
end

accuracy = sum / c;

disp('accuracy:');
disp(accuracy);

