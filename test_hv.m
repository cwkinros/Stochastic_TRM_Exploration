function [] = test_hv()

H = zeros(4);
y = 1;
w1 = 1;
w2 = 1;
n0 = 1;
n1 = 1;
n2 = 1;
inputs = 5;
outputs = 1;
bias1 = 0;
bias2 = 0;
n = 4;

h1 = w1*inputs;
g1 = sigmoid(h1);
g1_1 = sigmoid_1(h1);
g1_2 = sigmoid_2(h1);

h2 = g1*w2;
g2_1 = sigmoid_1(h2);
g2_2 = sigmoid_2(h2);

dg2 = sigmoid(h2) - y;
dg1 = dg2*sigmoid_1(h2)*w2;

e = dg2;
actualH = zeros(4,4);
actualH(3,3)= (sigmoid_1(h2)*g1)^2 + e*sigmoid_2(h2)*g1^2;
actualH(1,1) = ((sigmoid_1(h2))^2 + e*sigmoid_2(h2))*(w2*sigmoid_1(h1)*inputs)^2 + e*sigmoid_1(h2)*w2*sigmoid_2(h1)*inputs^2;
actualH(1,3) = ((sigmoid_1(h2))^2)*g1*w2*sigmoid_1(h1)*inputs + e*sigmoid_2(h2)*g1*w2*sigmoid_1(h1)*inputs + e*sigmoid_1(h2)*sigmoid_1(h1)*inputs;
actualH(3,1) = actualH(1,3);
actualH(4,4) = (sigmoid_1(h2))^2 + e*sigmoid_2(h2);
actualH(4,3) = (sigmoid_1(h2)^2)*g1 + e*sigmoid_2(h2)*g1;
actualH(3,4) = actualH(4,3);
actualH(4,1) = (sigmoid_1(h2)^2)*w2*sigmoid_1(h1)*inputs + e*sigmoid_2(h2)*w2*sigmoid_1(h1)*inputs;
actualH(1,4) = actualH(4,1);
actualH(2,2) = (sigmoid_1(h2)*w2*sigmoid_1(h1))^2 + e*sigmoid_2(h2)*(w2*sigmoid_1(h1))^2 + e*sigmoid_1(h2)*w2*sigmoid_2(h1);
actualH(2,3) = (sigmoid_1(h2)^2)*w2*g1*sigmoid_1(h1) + e*sigmoid_2(h2)*w2*g1*sigmoid_1(h1) + e*sigmoid_1(h2)*sigmoid_1(h1);
actualH(3,2) = actualH(2,3);
actualH(4,2) = (sigmoid_1(h2)^2)*w2*sigmoid_1(h1) + e*sigmoid_2(h2)*w2*sigmoid_1(h1);
actualH(2,4) = actualH(4,2);
actualH(1,2) = ((sigmoid_1(h2)*w2*sigmoid_1(h1))^2)*inputs + e*sigmoid_2(h2)*((w2*sigmoid_1(h1))^2)*inputs + e*sigmoid_1(h2)*w2*sigmoid_2(h1)*inputs;
actualH(2,1) = actualH(1,2);

%test1
I = eye(4);
for i = 1:4
    H(:,i) = Hv(I(:,i),w1,w2,g1,g1_1,g2_1,g2_2,g1_2,dg1,dg2,inputs,0);
end
if sum(sum(abs(H - actualH))) < 0.00000000001
    disp('Hv passed');
else
    disp('Hv failed');
end
for i = 1:4
    H(:,i) = Hv_WS(I(:,i),w1,w2,g1,g1_1,g2_1,g2_2,g1_2,dg1,dg2,inputs,0,[1,2,3,4],4);
end
if sum(sum(abs(H - actualH))) < 0.00000000001
    disp('Hv_WS passed');
else
    disp('Hv_WS failed');
end