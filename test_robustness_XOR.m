
n0 = 2;
n1 = 2;
n2 = 1;
testParams('XOR',[-5,1],[-5,1],[2,1],[2,1],[2,1],5000, false,false, n1)
% we want to test for robustness so this needs to happen many times
% we're going to write this custom based on run_test
tests = 'SGD MBGD TRM TRM_WS BTRM BTRM_WS MBTRM MBTRM_WS STRM STRM_WS';
tofile = true;
dataset = 'XOR';
for k = 1:100
    W1 = rand(n1, n0) - 0.5;
    W1 = W1/sum(sum(abs(W1)));
    W2 = rand(n2, n1) - 0.5;
    W2 = W2/sum(sum(abs(W2)));

    bias1 = rand(n1,1) - 0.5;
    bias2 = rand(n2,1) - 0.5;

    
    w = M1M2_to_m(W1,W2,bias1,bias2);
    file_weights = fopen(strcat(dataset,'_initialWeights',num2str(k),'.txt'),'w');
    fprintf(file_weights,'%f \n',w);
 
    for j = 1:10

        

        [~,len] = size(tests);
        i = 1;
        while i <= len
            start = i;
            while i <= len && tests(i) ~= ' '
                i = i + 1;
            end
            test = tests(start:i-1);
            i = i + 1;

            [WS,MS,TRMstep,GD,gamma] = getParams(test,lr,sgd_lr,b_m_mini);
            if tofile
                file = fopen(strcat('results/',dataset,'_',test,'weights',num2str(k),'trial',num2str(j),'.txt'),'w');
            else 
                file = 0;
            end
            [final_W1, final_W2, final_bias1, final_bias2, error] = train_TRM_united_w_param_control(WS,MS,TRMstep,GD,inputs, outputs, W1, W2, bias1, bias2, n1, maxiter, tofile, file, b_w, b_m_mini, b_m_big, gamma);
            print_accuracy2(inputs,outputs, final_W1, final_W2, final_bias1, final_bias2,tofile, file);
            w_final = M1M2_to_m(final_W1,final_W2,final_bias1,final_bias2);
            final_weights = fopen(strcat('results/',dataset,'_',test,'weights',num2str(k),'trial',num2str(j),'_finalw.txt'),'w');
            fprintf(final_weights,'%d \n',w_final);
        end
    end
end




