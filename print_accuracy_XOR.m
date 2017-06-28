function [] = print_accuracy_XOR(inputs,W1,W2, bias1, bias2, tofile, file)

[~,a] = size(inputs);
if isnan(bias1)
    h1 = W1*inputs;
    %disp(h1);
    g1 = sigmoid(h1);
    %disp(g1);
    outputs = sigmoid(W2*g1);   
else 
    h1 = W1*inputs + bias1*ones(1,a);
    %disp(h1);
    g1 = sigmoid(h1);
    %disp(g1);
    outputs = sigmoid(W2*g1 + bias2*ones(1,a));
end
[r,c] = size(outputs);
sum = 0;
%disp(outputs);
if r == 1
    if outputs(1) < 0.5
        sum = sum + 1;
    end
    if outputs(2) > 0.5
        sum = sum + 1;
    end
    if outputs(3) > 0.5
        sum = sum + 1;
    end
    if outputs(4) < 0.5
        sum = sum + 1;
    end
else
    disp('whattt');
end


accuracy = sum / c;
if tofile
    fprintf(file, '%f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0', accuracy);
else
    disp('accuracy:');  
    disp(accuracy);
end