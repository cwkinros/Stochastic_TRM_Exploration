function [m] = M1M2_to_m(M1,M2,bias1,bias2)

[r1,c1] = size(M1);
[r2,c2] = size(M2);

m = zeros(r1*c1 + r2*c2,1);
count = 1;
for i = 1:r1
    for j = 1:c1
        m(count) = M1(i,j);
        count = count + 1;
    end
    m(count) = bias1(i);
    count = count + 1;
end

for i = 1:r2
    for j = 1:c2
        m(count) = M2(i,j);
        count = count + 1;
    end
    m(count) = bias2(i);
    count = count + 1;
end
