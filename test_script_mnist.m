

% for running tests on MNIST
maxiter = 100000;
testParams('MNIST',false,false,false,[20,20],false,[-3,1],[-3,1],[20,20],maxiter,true, 10);


maxiter = 100000;
run_w_params('Nurs','MBGD TRM_MBGD MBTRM_WS',maxiter,0, true);
run_w_params('Nurs','MBTRM',maxiter,0,false);






