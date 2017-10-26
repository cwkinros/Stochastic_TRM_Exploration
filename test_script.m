

% for running tests on MNIST
%maxiter = 10000;
%testParams('MNIST',false,false,false,[20,20],false,[-3,1],[-3,1],[20,20],maxiter,true, 10);

maxiter = 1000;
run_w_params_plus_test('MNIST','TRM',maxiter,1);
run_w_params_plus_test('MNIST','TRM',maxiter,2);
run_w_params_plus_test('MNIST','TRM',maxiter,3);
run_w_params_plus_test('MNIST','TRM',maxiter,4);
run_w_params_plus_test('MNIST','TRM',maxiter,5);







