function [gmag,error] = testTRM()

w1 = 0.5;
w2 = 0.1;
bias1 = 0;
bias2 = 0;
n0 = 1;
n1 = 1;
n2 = 1;
n = 4;

inputs = 5;
outputs = 0.5;
%[~,~,~,error] = train_GD(inputs,outputs,w1,w2,bias1,bias2);
[~,~,~,~,~,~,~,gmag] = train_TRM(inputs,outputs,w1,w2,bias1,bias2,n1,100);
