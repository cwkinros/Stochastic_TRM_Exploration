function [inputs,outputs] = createSyntheticData(m)

%equation used is y = 4x + 2 (but with 0.4 random noise)

noise = 1;
inputs = rand(2,m)*10;
outputs = zeros(1,m);
%figure;
for i = 1:m
    if 2*inputs(1,i) - inputs(2,i) > (5 + normrnd(0,noise))
        outputs(1,i) = 1;
        %plot(inputs(1,i),inputs(2,i),'.','color','r');
    else
        outputs(1,i) = 0;
        %plot(inputs(1,i),inputs(2,i),'.','color','b');
    end
    %hold on;
end





