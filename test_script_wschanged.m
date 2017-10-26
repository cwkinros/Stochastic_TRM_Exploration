

% for running tests on Derm
maxiter = 200000;
run_w_params('Derm','MBTRM_WS STRM_WS',maxiter);

run_w_params('IRIS','MBTRM_WS STRM_WS',maxiter);

run_w_params('Habe','MBTRM_WS STRM_WS',maxiter);

