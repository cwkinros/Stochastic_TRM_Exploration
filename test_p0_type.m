   

[inputs,outputs] = getIrisData();
b_w = 0;
b_m_mini = 0;
b_m_big = 0;
gamma = 1;
tofile = true;
maxiter = 1000;
[n0,m] = size(inputs);
n1 = 10;
[n2,~] = size(outputs);
[W1,W2,bias1,bias2] = getWeightsFromFile('IRIS_initialWeights.txt',n0,n1,n2);

WS = false;
MS = 0;
TRMstep = true;
GD = false;


for i = 1:4
    for tries = 1:3
        file = fopen(strcat('results/test_p0_type',int2str(i),'_trial',int2str(tries),'.txt'),'w');
        [final_W1, final_W2, final_bias1, final_bias2, error] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, gamma,i);
        print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2,tofile, file);
        fclose(file);
    end
end