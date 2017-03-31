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
lambda = 5;
m=1;

[g,g1,g1_1,g2_1,g2_2,g1_2,dg1,dg2,error] = getG(w1,w2,bias1,bias2,inputs,outputs,lambda,m);
e = dg2;
actualG = zeros(4,1);
actualG(1) = e*g2_1*w2*g1_1*inputs + w1*lambda;
actualG(2) = e*g2_1*w2*g1_1 + bias1*lambda;
actualG(3) = e*g2_1*g1 + w2*lambda;
actualG(4) = e*g2_1 + bias2*lambda;


actualH = zeros(4,4);
actualH(3,3)= (g2_1*g1)^2 + e*g2_2*g1^2;
actualH(1,1) = ((g2_1)^2 + e*g2_2)*(w2*g1_1*inputs)^2 + e*g2_1*w2*g1_2*inputs^2;
actualH(1,3) = ((g2_1)^2)*g1*w2*g1_1*inputs + e*g2_2*g1*w2*g1_1*inputs + e*g2_1*g1_1*inputs;
actualH(3,1) = actualH(1,3);
actualH(4,4) = (g2_1)^2 + e*g2_2;
actualH(4,3) = (g2_1^2)*g1 + e*g2_2*g1;
actualH(3,4) = actualH(4,3);
actualH(4,1) = (g2_1^2)*w2*g1_1*inputs + e*g2_2*w2*g1_1*inputs;
actualH(1,4) = actualH(4,1);
actualH(2,2) = (g2_1*w2*g1_1)^2 + e*g2_2*(w2*g1_1)^2 + e*g2_1*w2*g1_2;
actualH(2,3) = (g2_1^2)*w2*g1*g1_1 + e*g2_2*w2*g1*g1_1 + e*g2_1*g1_1;
actualH(3,2) = actualH(2,3);
actualH(4,2) = (g2_1^2)*w2*g1_1 + e*g2_2*w2*g1_1;
actualH(2,4) = actualH(4,2);
actualH(1,2) = ((g2_1*w2*g1_1)^2)*inputs + e*g2_2*((w2*g1_1)^2)*inputs + e*g2_1*w2*g1_2*inputs;
actualH(2,1) = actualH(1,2);

%test1

I = eye(4);
for i = 1:4
    H(:,i) = Hv(I(:,i),w1,w2,g1,g1_1,g2_1,g2_2,g1_2,dg1,dg2,inputs,lambda);
end

actualH = actualH + lambda*eye(n);
if sum(sum(abs(H - actualH))) < 0.00000000001
    disp('Hv passed');
else
    disp('Hv failed');
end


for i = 1:4
    H(:,i) = Hv_WS(I(:,i),w1,w2,g1,g1_1,g2_1,g2_2,g1_2,dg1,dg2,inputs,lambda,[1,2,3,4],4);
end
if sum(sum(abs(H - actualH))) < 0.00000000001
    disp('Hv_WS passed');
else
    disp('Hv_WS failed');
end

if sum(abs(g - actualG)) < 0.00000000001
    disp('G passed');
else 
    disp('G failed');
end

