function [M1,M2,bias1,bias2] = m_to_M1M2(m,n0,n1,n2)

r1 = n1;
c1 = n0;

r2 = n2;
c2 = n1;

M1 = zeros(r1,c1);
bias1 = zeros(r1,1);

M2 = zeros(r2,c2);
bias2 = zeros(r2,1);
count = 1;
for i = 1:r1
    for j = 1:c1
        M1(i,j) = m(count);
        count = count + 1;
    end
    bias1(i) = m(count);
    count = count + 1;
end

for i = 1:r2
    for j = 1:c2
        M2(i,j) = m(count);
        count = count + 1;
    end
    bias2(i) = m(count);
    count = count + 1;
end