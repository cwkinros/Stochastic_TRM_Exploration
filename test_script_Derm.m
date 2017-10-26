

% for running tests on Derm
maxiter = 100000;
testParams('Derm',false,false,[10,10],false,[50,50],5000,true,10);
run_w_params('Derm','TRM TRM_WS BTRM BTRM_WS',maxiter);
return;
testParams('IRIS',false,false,[5,5],false,[30,30],5000,true,10);
run_w_params('IRIS','TRM TRM_WS BTRM BTRM_WS',maxiter);

testParams('Habe',false,false,[5,5],false,[50,50],5000,true,10);
run_w_params('Habe','TRM TRM_WS BTRM BTRM_WS',maxiter);