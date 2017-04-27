

% for running tests on Derm
testParams('IRIS',[-4,1],[-4,1],[5,5],[5,5],[50,50],5000, false)
run_w_params('IRIS','SGD MBGD TRM TRM_WS BTRM BTRM_WS MBTRM MBTRM_WS STRM STRM_WS');