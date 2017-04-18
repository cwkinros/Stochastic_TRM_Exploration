function [sub_inputs] = downsample_mnist(inputs)

[~,m] = size(inputs);

square_inputs = zeros(28,28,m);

for i = 1:m
    square_inputs(:,:,i) = reshape(inputs(:,i),28,28);
    image(255*square_inputs(:,:,i));
end

square_sub_inputs = zeros(10,10,m);


vals = [4,4,2,2,2,2,2,2,4,4];
ms = vals.'*vals;
row = 1;
col = 1;
for k = 1:m
    row = 1;
    for i = 1:10
        col = 1;
        for j = 1:10
            total = sum(sum(square_inputs(row:row+vals(i) - 1, col:col+vals(j) - 1,k)));
            square_sub_inputs(i,j,k) = total / ms(i,j);
            col = col + vals(j);
        end
        row = row + vals(i);
    end
end

sub_inputs = zeros(100,m);
for i = 1:m
    sub_inputs(:,i) = reshape(square_sub_inputs(:,:,i),100,1);
end