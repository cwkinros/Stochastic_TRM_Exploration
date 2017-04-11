

r1 = readtable('test_p0_method1.txt');
r2 = readtable('test_p0_method2.txt');
r3 = readtable('test_p0_method3.txt');
r4 = readtable('test_p0_method4.txt');

[n1,~] = size(r1.time);
[n2,~] = size(r2.time);
[n3,~] = size(r3.time);
[n4,~] = size(r4.time);



r1_t = zeros(n1,1);
r2_t = zeros(n2,1);
r3_t = zeros(n3,1);
r4_t = zeros(n4,1);

r1_t(1) = r1.time(1);
r2_t(1) = r2.time(1);
r3_t(1) = r3.time(1);
r4_t(1) = r4.time(1);
for i=2:n1
    r1_t(i) = r1_t(i-1) + r1.time(i);
end

for i=2:n2
    r2_t(i) = r2_t(i-1) + r2.time(i);
end

for i=2:n3
    r3_t(i) = r3_t(i-1) + r3.time(i);
end

for i=2:n4
    r3_t(i) = r3_t(i-1) + r4.time(i);
end
figure;
plot(r1_t,r1.totalError);
hold on;
plot(r2_t,r2.totalError);
hold on;
plot(r3_t,r3.totalError);
hold on;
plot(r4_t,r4.totalError);
legend('pcg','cgs','minres','lsqr');

