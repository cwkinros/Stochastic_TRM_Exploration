

% for running tests on Derm
maxiter = 200000;
testParams('Nurs',false,false,[10,10],false,[500,500],5000,true,10);
run_w_params('Nurs','TRM_WS BTRM BTRM_WS',maxiter);

