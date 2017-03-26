function [full_error] = get_full_error(inputs,outputs,W1,W2,bias1,bias2,m)

 h1s = W1*inputs + bias1*ones(1,m);
g1s = sigmoid(h1s);

h2s = W2*g1s + bias2*ones(1,m);
g2s = sigmoid(h2s);

errors = (g2s - outputs);
full_error = sum(sum(errors.*errors));