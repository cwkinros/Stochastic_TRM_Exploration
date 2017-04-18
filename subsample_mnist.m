function [sub_inputs] = downsample_mnist(inputs)

[~,m] = size(inputs);

square_inputs = zeros(28,28,m);
for i = 1:m
    square_inputs(:,:,i) = reshape(inputs(:,i),28,28);
    image(255*square_inputs(:,:,i));
end

sub_inputs = zeros(10,10,m);


vals = [4,4,2,2,2,2,2,2,4,4];
m = vals.'*vals;
row = 0;
for k = 1:m
    for i = 1:10
        for j = 1:10
            total = sum(sum(square_inputs(row:row+vals(i) - 1, col:col+vals(j) - 1,k)));
            sub_inputs(i,j,k) = total / m(i,j);
            col = col + vals(i);
        end
        row = row + vals(i);
    end
end

image(255*sub_inputs(:,:,1));