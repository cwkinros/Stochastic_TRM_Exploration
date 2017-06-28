function [] = Overfitting_experiment()

n0 = 2;
n1 = 10;
n2 = 1;

W1 = rand(n1,n0);
bias1 = rand(n1,1);
W2 = rand(n2,n1);
bias2 = rand(n2,1);
image_it = zeros(500,500,3);
[rows,columns,~] = size(image_it);

incr = 10;
[inputs,outputs] = createSyntheticData(incr);

m = incr;
for i = 1:1000
    [w1,w2,b1,b2,~,~] = train_TRM_united_w_param_control(0,0,true,0,inputs,outputs,W1,W2,bias1,bias2,n1,1000,false,0,0,0,0,0,0,[],[],true,60*10);
    figure;
    subplot(2,1,1);
    for r = 1:rows
        for c = 1:columns
            val = sigmoid(w2*sigmoid(w1*[c/(rows/10);r/(rows/10)] + b1) + b2);
            image_it(rows + 1 - r,c,1) = val;
            image_it(rows + 1 - r,c,3) = 1-val;
        end
    end  
    image(image_it);
    title('Learned Classification Map');
    
    subplot(2,1,2);
    for k = 1:m
        if outputs(1,k) == 1
            plot(inputs(1,k),inputs(2,k),'.','color','r');
        else
            plot(inputs(1,k),inputs(2,k),'.','color','b');
        end
        hold on;
    end
    title(strcat('Training Samples: m = ',num2str(m))); 
    fig = gcf;
    print(fig,strcat('m_',num2str(m)),'-dpng','-r0');
    
    old_m = m;
    old_inputs = inputs;
    old_outputs = outputs;
    [inputs_xtra,outputs_xtra] = createSyntheticData(incr);
    m = m + incr;
    inputs = zeros(2,m);
    outputs = zeros(1,m);
    inputs(:,1:old_m) = old_inputs;
    inputs(:,old_m+1:m) = inputs_xtra;
    outputs(1,1:old_m) = old_outputs;
    outputs(1,old_m+1:m) = outputs_xtra;
end
    
    