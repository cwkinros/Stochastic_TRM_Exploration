[inputs,outputs] = getHabermanData();
[~,m] = size(inputs);
figure;
for i = 1:m
    if outputs(i)
        scatter3(inputs(1,i),inputs(2,i), inputs(3,i),'*','MarkerEdgeColor','b');
    else
        scatter3(inputs(1,i),inputs(2,i), inputs(3,i),'o','MarkerEdgeColor','r');
    end
    hold on;
end