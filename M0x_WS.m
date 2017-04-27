function [m0x] = M0x_WS(x, g, gamma, W1, W2, g1s, g1_1s, g2_1s, g2_2s, g1_2s, dg1s, dg2s, inputs, indices, lambda)

[two_n,m] = size(x);

if (m > two_n)
    two_n = m;
    x = x.';
end

[n1,n0] = size(W1);
[n2,~] = size(W2);
n_w_bias = n2*(n1 + 1) + n1*(n0 + 1);

n = two_n / 2;
if n < n_w_bias
    bias = false;
else
    bias = true;
end
m0x = zeros(two_n,1);

m0x(1:n) = -x(1:n) + Hv_WS(x(n+1:two_n),W1, W2, g1s, g1_1s, g2_1s, g2_2s, g1_2s, dg1s, dg2s, inputs,lambda, indices, n,bias);
m0x(n+1:two_n) = Hv_WS(x(1:n),W1, W2,g1s, g1_1s, g2_1s, g2_2s, g1_2s, dg1s, dg2s,inputs, lambda, indices, n,bias)- g*(g.'*x(n+1:two_n)) / gamma^2;